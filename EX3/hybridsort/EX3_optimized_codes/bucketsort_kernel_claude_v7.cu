#ifndef _BUCKETSORT_KERNEL_H_
#define _BUCKETSORT_KERNEL_H_

#include <stdio.h>

#define BUCKET_WARP_LOG_SIZE 5
#define BUCKET_WARP_N 1
#ifdef BUCKET_WG_SIZE_1
#define BUCKET_THREAD_N BUCKET_WG_SIZE_1
#else
#define BUCKET_THREAD_N (BUCKET_WARP_N << BUCKET_WARP_LOG_SIZE)
#endif
#define BUCKET_BLOCK_MEMORY (DIVISIONS * BUCKET_WARP_N)
#define BUCKET_BAND 128

__device__ __forceinline__ int addOffset(volatile unsigned int *s_offset, unsigned int data,
                         unsigned int threadTag) {
    unsigned int count;
    unsigned int old_val;

    #pragma unroll 2
    do {
        old_val = s_offset[data];
        count = old_val & 0x07FFFFFFU;
        count = threadTag | (count + 1);
        s_offset[data] = count;
    } while (s_offset[data] != count);

    return (count & 0x07FFFFFFU) - 1;
}

__global__ void __launch_bounds__(256, 4) bucketcount(float *input, int *indice,
                            unsigned int *d_prefixoffsets, int size,
                            cudaTextureObject_t texPivot) {
    volatile __shared__ unsigned int s_offset[BUCKET_BLOCK_MEMORY];

    const unsigned int threadTag = threadIdx.x << (32 - BUCKET_WARP_LOG_SIZE);
    const int warpBase = (threadIdx.x >> BUCKET_WARP_LOG_SIZE) * DIVISIONS;
    const int numThreads = blockDim.x * gridDim.x;
    const int tid_start = blockIdx.x * blockDim.x + threadIdx.x;
    
    #pragma unroll 4
    for (int i = threadIdx.x; i < BUCKET_BLOCK_MEMORY; i += blockDim.x)
        s_offset[i] = 0;

    __syncthreads();

    const int grid_stride = numThreads * 4;
    for (int tid = tid_start; tid < size; tid += grid_stride) {
        float elem1, elem2, elem3, elem4;
        int idx1, idx2, idx3, idx4;
        
        elem1 = (tid < size) ? __ldg(&input[tid]) : 0.0f;
        elem2 = (tid + numThreads < size) ? __ldg(&input[tid + numThreads]) : 0.0f;
        elem3 = (tid + 2*numThreads < size) ? __ldg(&input[tid + 2*numThreads]) : 0.0f;
        elem4 = (tid + 3*numThreads < size) ? __ldg(&input[tid + 3*numThreads]) : 0.0f;

        if (tid < size) {
            idx1 = DIVISIONS / 2 - 1;
            int jump = DIVISIONS / 4;
            float piv = tex1Dfetch<float>(texPivot, idx1);

            #pragma unroll
            while (jump >= 1) {
                idx1 = (elem1 < piv) ? (idx1 - jump) : (idx1 + jump);
                piv = tex1Dfetch<float>(texPivot, idx1);
                jump /= 2;
            }
            idx1 = (elem1 < piv) ? idx1 : (idx1 + 1);

            indice[tid] =
                (addOffset(s_offset + warpBase, idx1, threadTag) << LOG_DIVISIONS) + idx1;
        }

        if (tid + numThreads < size) {
            idx2 = DIVISIONS / 2 - 1;
            int jump = DIVISIONS / 4;
            float piv = tex1Dfetch<float>(texPivot, idx2);

            #pragma unroll
            while (jump >= 1) {
                idx2 = (elem2 < piv) ? (idx2 - jump) : (idx2 + jump);
                piv = tex1Dfetch<float>(texPivot, idx2);
                jump /= 2;
            }
            idx2 = (elem2 < piv) ? idx2 : (idx2 + 1);

            indice[tid + numThreads] =
                (addOffset(s_offset + warpBase, idx2, threadTag) << LOG_DIVISIONS) + idx2;
        }

        if (tid + 2*numThreads < size) {
            idx3 = DIVISIONS / 2 - 1;
            int jump = DIVISIONS / 4;
            float piv = tex1Dfetch<float>(texPivot, idx3);

            #pragma unroll
            while (jump >= 1) {
                idx3 = (elem3 < piv) ? (idx3 - jump) : (idx3 + jump);
                piv = tex1Dfetch<float>(texPivot, idx3);
                jump /= 2;
            }
            idx3 = (elem3 < piv) ? idx3 : (idx3 + 1);

            indice[tid + 2*numThreads] =
                (addOffset(s_offset + warpBase, idx3, threadTag) << LOG_DIVISIONS) + idx3;
        }

        if (tid + 3*numThreads < size) {
            idx4 = DIVISIONS / 2 - 1;
            int jump = DIVISIONS / 4;
            float piv = tex1Dfetch<float>(texPivot, idx4);

            #pragma unroll
            while (jump >= 1) {
                idx4 = (elem4 < piv) ? (idx4 - jump) : (idx4 + jump);
                piv = tex1Dfetch<float>(texPivot, idx4);
                jump /= 2;
            }
            idx4 = (elem4 < piv) ? idx4 : (idx4 + 1);

            indice[tid + 3*numThreads] =
                (addOffset(s_offset + warpBase, idx4, threadTag) << LOG_DIVISIONS) + idx4;
        }
    }

    __syncthreads();

    int prefixBase = blockIdx.x * BUCKET_BLOCK_MEMORY;

    #pragma unroll 4
    for (int i = threadIdx.x; i < BUCKET_BLOCK_MEMORY; i += blockDim.x)
        d_prefixoffsets[prefixBase + i] = s_offset[i] & 0x07FFFFFFU;
}

__global__ void __launch_bounds__(256, 4) bucketprefixoffset(unsigned int *d_prefixoffsets,
                                   unsigned int *d_offsets, int blocks) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blocks * BUCKET_BLOCK_MEMORY;
    unsigned int sum = 0;

    #pragma unroll 4
    for (int i = tid; i < size; i += DIVISIONS) {
        unsigned int x = __ldg(&d_prefixoffsets[i]);
        d_prefixoffsets[i] = sum;
        sum += x;
    }

    d_offsets[tid] = sum;
}

__global__ void __launch_bounds__(256, 4) bucketsort(float *input, int *indice, float *output, int size,
                           unsigned int *d_prefixoffsets,
                           unsigned int *l_offsets) {
    volatile __shared__ unsigned int s_offset[BUCKET_BLOCK_MEMORY];

    int prefixBase = blockIdx.x * BUCKET_BLOCK_MEMORY;
    const int warpBase = (threadIdx.x >> BUCKET_WARP_LOG_SIZE) * DIVISIONS;
    const int numThreads = blockDim.x * gridDim.x;
    
    #pragma unroll 4
    for (int i = threadIdx.x; i < BUCKET_BLOCK_MEMORY; i += blockDim.x) {
        s_offset[i] = __ldg(&l_offsets[i & (DIVISIONS - 1)]) + __ldg(&d_prefixoffsets[prefixBase + i]);
    }

    __syncthreads();

    const int tid_start = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_stride = numThreads * 4;
    
    for (int tid = tid_start; tid < size; tid += grid_stride) {
        if (tid < size) {
            float elem = __ldg(&input[tid]);
            int id = __ldg(&indice[tid]);
            int offset = s_offset[warpBase + (id & (DIVISIONS - 1))] + (id >> LOG_DIVISIONS);
            output[offset] = elem;
        }

        if (tid + numThreads < size) {
            float elem = __ldg(&input[tid + numThreads]);
            int id = __ldg(&indice[tid + numThreads]);
            int offset = s_offset[warpBase + (id & (DIVISIONS - 1))] + (id >> LOG_DIVISIONS);
            output[offset] = elem;
        }

        if (tid + 2*numThreads < size) {
            float elem = __ldg(&input[tid + 2*numThreads]);
            int id = __ldg(&indice[tid + 2*numThreads]);
            int offset = s_offset[warpBase + (id & (DIVISIONS - 1))] + (id >> LOG_DIVISIONS);
            output[offset] = elem;
        }

        if (tid + 3*numThreads < size) {
            float elem = __ldg(&input[tid + 3*numThreads]);
            int id = __ldg(&indice[tid + 3*numThreads]);
            int offset = s_offset[warpBase + (id & (DIVISIONS - 1))] + (id >> LOG_DIVISIONS);
            output[offset] = elem;
        }
    }
}

#endif
