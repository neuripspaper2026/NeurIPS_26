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

// Removed deprecated texture declaration
// texture<float, 1, cudaReadModeElementType> texPivot;

__device__ __forceinline__ int addOffset(volatile unsigned int *s_offset, unsigned int data,
                         unsigned int threadTag) {
    unsigned int count;
    volatile unsigned int *ptr = s_offset + data;

#pragma unroll 1
    do {
        count = *ptr & 0x07FFFFFFU;
        count = threadTag | (count + 1);
        *ptr = count;
    } while (*ptr != count);

    return (count & 0x07FFFFFFU) - 1;
}

__global__ void bucketcount(float *input, int *indice,
                            unsigned int *d_prefixoffsets, int size,
                            cudaTextureObject_t texPivot) {
    extern __shared__ unsigned int shmem[];
    volatile unsigned int *s_offset = shmem; // size: BUCKET_BLOCK_MEMORY

    const unsigned int threadTag = static_cast<unsigned int>(threadIdx.x) << (32 - BUCKET_WARP_LOG_SIZE);
    const int warpBase = (threadIdx.x >> BUCKET_WARP_LOG_SIZE) * DIVISIONS;
    const int numThreads = blockDim.x * gridDim.x;

#pragma unroll
    for (int i = threadIdx.x; i < BUCKET_BLOCK_MEMORY; i += blockDim.x) {
        s_offset[i] = 0;
    }

    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll 1
    for (; tid < size; tid += numThreads) {
        float elem = __ldg(&input[tid]);

        int idx = DIVISIONS / 2 - 1;
        int jump = DIVISIONS / 4;
        float piv = tex1Dfetch<float>(texPivot, idx);

#pragma unroll
        while (jump >= 1) {
            idx = (elem < piv) ? (idx - jump) : (idx + jump);
            piv = tex1Dfetch<float>(texPivot, idx);
            jump >>= 1;
        }
        idx = (elem < piv) ? idx : (idx + 1);

        int local = addOffset((unsigned int *)(s_offset + warpBase), static_cast<unsigned int>(idx), threadTag);
        indice[tid] = (local << LOG_DIVISIONS) + idx;
    }

    __syncthreads();

    int prefixBase = blockIdx.x * BUCKET_BLOCK_MEMORY;

#pragma unroll
    for (int i = threadIdx.x; i < BUCKET_BLOCK_MEMORY; i += blockDim.x) {
        d_prefixoffsets[prefixBase + i] = s_offset[i] & 0x07FFFFFFU;
    }
}

__global__ void bucketprefixoffset(unsigned int *d_prefixoffsets,
                                   unsigned int *d_offsets, int blocks) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blocks * BUCKET_BLOCK_MEMORY;
    unsigned int sum = 0;

#pragma unroll 1
    for (int i = tid; i < size; i += DIVISIONS) {
        unsigned int x = d_prefixoffsets[i];
        d_prefixoffsets[i] = sum;
        sum += x;
    }

    d_offsets[tid] = sum;
}

__global__ void bucketsort(float *input, int *indice, float *output, int size,
                           unsigned int *d_prefixoffsets,
                           unsigned int *l_offsets) {
    extern __shared__ unsigned int shmem[];
    volatile unsigned int *s_offset = shmem; // size: BUCKET_BLOCK_MEMORY

    int prefixBase = blockIdx.x * BUCKET_BLOCK_MEMORY;
    const int warpBase = (threadIdx.x >> BUCKET_WARP_LOG_SIZE) * DIVISIONS;
    const int numThreads = blockDim.x * gridDim.x;

#pragma unroll
    for (int i = threadIdx.x; i < BUCKET_BLOCK_MEMORY; i += blockDim.x) {
        unsigned int bucket = static_cast<unsigned int>(i) & (DIVISIONS - 1);
        s_offset[i] = l_offsets[bucket] + d_prefixoffsets[prefixBase + i];
    }

    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll 1
    for (; tid < size; tid += numThreads) {
        float elem = __ldg(&input[tid]);
        int id = indice[tid];
        int bucket = id & (DIVISIONS - 1);
        int intra = id >> LOG_DIVISIONS;
        unsigned int base = s_offset[warpBase + bucket];
        output[base + intra] = elem;
        int test = base + intra;
        (void)test;
    }
}

#endif
