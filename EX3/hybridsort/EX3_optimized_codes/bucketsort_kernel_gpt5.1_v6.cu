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

__device__ __forceinline__ int addOffset(volatile unsigned int *s_offset,
                                         unsigned int data,
                                         unsigned int threadTag) {
    unsigned int count;

#pragma unroll 8
    do {
        count = s_offset[data] & 0x07FFFFFFU;
        count = threadTag | (count + 1);
        s_offset[data] = count;
    } while (s_offset[data] != count);

    return (count & 0x07FFFFFFU) - 1;
}

__global__ void bucketcount(float * __restrict__ input,
                            int   * __restrict__ indice,
                            unsigned int * __restrict__ d_prefixoffsets,
                            int size,
                            cudaTextureObject_t texPivot) {
    extern __shared__ unsigned int s_mem[];
    volatile unsigned int * __restrict__ s_offset = s_mem;

    const unsigned int threadTag =
        static_cast<unsigned int>(threadIdx.x) << (32 - BUCKET_WARP_LOG_SIZE);
    const int warpBase = (threadIdx.x >> BUCKET_WARP_LOG_SIZE) * DIVISIONS;
    const int numThreads = blockDim.x * gridDim.x;

    // Initialize shared offsets
#pragma unroll
    for (int i = threadIdx.x; i < BUCKET_BLOCK_MEMORY; i += blockDim.x) {
        s_offset[i] = 0;
    }

    __syncthreads();

    // Grid-stride loop for better load balancing and full device utilization
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < size;
         tid += numThreads) {
        float elem = __ldg(&input[tid]);

        // Binary search over pivots (kept as iso-binary search)
        int idx  = DIVISIONS / 2 - 1;
        int jump = DIVISIONS / 4;
        float piv = tex1Dfetch<float>(texPivot, idx);

#pragma unroll
        while (jump >= 1) {
            idx = (elem < piv) ? (idx - jump) : (idx + jump);
            piv = tex1Dfetch<float>(texPivot, idx);
            jump >>= 1;
        }
        idx = (elem < piv) ? idx : (idx + 1);

        int offsetVal =
            (addOffset(s_offset + warpBase, static_cast<unsigned int>(idx),
                       threadTag)
             << LOG_DIVISIONS) +
            idx;

        indice[tid] = offsetVal;
    }

    __syncthreads();

    const int prefixBase = blockIdx.x * BUCKET_BLOCK_MEMORY;

#pragma unroll
    for (int i = threadIdx.x; i < BUCKET_BLOCK_MEMORY; i += blockDim.x) {
        d_prefixoffsets[prefixBase + i] = s_offset[i] & 0x07FFFFFFU;
    }
}

__global__ void bucketprefixoffset(unsigned int * __restrict__ d_prefixoffsets,
                                   unsigned int * __restrict__ d_offsets,
                                   int blocks) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blocks * BUCKET_BLOCK_MEMORY;
    int sum = 0;

    // Process every DIVISIONS-th element to keep writes coalesced
    for (int i = tid; i < size; i += DIVISIONS) {
        int x = d_prefixoffsets[i];
        d_prefixoffsets[i] = sum;
        sum += x;
    }

    d_offsets[tid] = sum;
}

__global__ void bucketsort(float * __restrict__ input,
                           int   * __restrict__ indice,
                           float * __restrict__ output,
                           int size,
                           unsigned int * __restrict__ d_prefixoffsets,
                           unsigned int * __restrict__ l_offsets) {
    extern __shared__ unsigned int s_mem[];
    volatile unsigned int * __restrict__ s_offset = s_mem;

    const int prefixBase = blockIdx.x * BUCKET_BLOCK_MEMORY;
    const int warpBase = (threadIdx.x >> BUCKET_WARP_LOG_SIZE) * DIVISIONS;
    const int numThreads = blockDim.x * gridDim.x;

    // Precompute per-bucket offsets in shared memory
#pragma unroll
    for (int i = threadIdx.x; i < BUCKET_BLOCK_MEMORY; i += blockDim.x) {
        unsigned int bucket = static_cast<unsigned int>(i) & (DIVISIONS - 1);
        s_offset[i] = l_offsets[bucket] + d_prefixoffsets[prefixBase + i];
    }

    __syncthreads();

    // Grid-stride loop for scatter
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < size;
         tid += numThreads) {

        float elem = __ldg(&input[tid]);
        int id = indice[tid];

        unsigned int bucket = static_cast<unsigned int>(id) & (DIVISIONS - 1);
        unsigned int indexInBucket =
            static_cast<unsigned int>(id) >> LOG_DIVISIONS;

        unsigned int base = s_offset[warpBase + bucket];
        unsigned int outPos = base + indexInBucket;

        output[outPos] = elem;

        // Preserve original computation (even if unused) for consistency
        int test = static_cast<int>(outPos);
        (void)test;
    }
}

#endif
