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
    unsigned int old;

    #pragma unroll 1
    do {
        old = s_offset[data];
        count = old & 0x07FFFFFFU;
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
    
    #pragma unroll 4
    for (int i = threadIdx.x; i < BUCKET_BLOCK_MEMORY; i += blockDim.x)
        s_offset[i] = 0;

    __syncthreads();

    const int tid_start = blockIdx.x * blockDim.x + threadIdx.x;
    
    #pragma unroll 2
    for (int tid = tid_start; tid < size; tid += numThreads) {
        float elem = __ldg(&input[tid]);

        int idx = DIVISIONS / 2 - 1;
        int jump = DIVISIONS / 4;
        float piv = tex1Dfetch<float>(texPivot, idx);

        #pragma unroll
        while (jump >= 1) {
            idx = (elem < piv) ? (idx - jump) : (idx + jump);
            piv = tex1Dfetch<float>(texPivot, idx);
            jump /= 2;
        }
        idx = (elem < piv) ? idx : (idx + 1);

        int offset = addOffset(s_offset + warpBase, idx, threadTag);
        indice[tid] = (offset << LOG_DIVISIONS) + idx;
    }

    __syncthreads();

    const int prefixBase = blockIdx.x * BUCKET_BLOCK_MEMORY;

    #pragma unroll 4
    for (int i = threadIdx.x; i < BUCKET_BLOCK_MEMORY; i += blockDim.x)
        d_prefixoffsets[prefixBase + i] = s_offset[i] & 0x07FFFFFFU;
}

__global__ void __launch_bounds__(256, 4) bucketprefixoffset(unsigned int *d_prefixoffsets,
                                   unsigned int *d_offsets, int blocks) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blocks * BUCKET_BLOCK_MEMORY;
    unsigned int sum = 0;

    #pragma unroll 4
    for (int i = tid; i < size; i += DIVISIONS) {
        unsigned int x = d_prefixoffsets[i];
        d_prefixoffsets[i] = sum;
        sum += x;
    }

    d_offsets[tid] = sum;
}

__global__ void __launch_bounds__(256, 4) bucketsort(float *input, int *indice, float *output, int size,
                           unsigned int *d_prefixoffsets,
                           unsigned int *l_offsets) {
    volatile __shared__ unsigned int s_offset[BUCKET_BLOCK_MEMORY];

    const int prefixBase = blockIdx.x * BUCKET_BLOCK_MEMORY;
    const int warpBase = (threadIdx.x >> BUCKET_WARP_LOG_SIZE) * DIVISIONS;
    const int numThreads = blockDim.x * gridDim.x;
    
    #pragma unroll 4
    for (int i = threadIdx.x; i < BUCKET_BLOCK_MEMORY; i += blockDim.x) {
        const int offset_idx = i & (DIVISIONS - 1);
        s_offset[i] = __ldg(&l_offsets[offset_idx]) + __ldg(&d_prefixoffsets[prefixBase + i]);
    }

    __syncthreads();

    const int tid_start = blockIdx.x * blockDim.x + threadIdx.x;

    #pragma unroll 2
    for (int tid = tid_start; tid < size; tid += numThreads) {
        float elem = __ldg(&input[tid]);
        int id = __ldg(&indice[tid]);

        const int bucket_idx = id & (DIVISIONS - 1);
        const int offset_in_bucket = id >> LOG_DIVISIONS;
        const int output_pos = s_offset[warpBase + bucket_idx] + offset_in_bucket;
        
        output[output_pos] = elem;
    }
}

#endif
