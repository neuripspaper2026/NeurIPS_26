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

    do {
        count = s_offset[data] & 0x07FFFFFFU;
        count = threadTag | (count + 1);
        s_offset[data] = count;
        // ensure visibility within the warp/block for the updated value
        __threadfence_block();
    } while (s_offset[data] != count);

    return (count & 0x07FFFFFFU) - 1;
}

__global__ void __launch_bounds__(BUCKET_THREAD_N, 2)
bucketcount(float * __restrict__ input,
            int * __restrict__ indice,
            unsigned int * __restrict__ d_prefixoffsets,
            int size,
            cudaTextureObject_t texPivot) {
    extern __shared__ unsigned int s_offset_dyn[];
    volatile unsigned int *s_offset = s_offset_dyn;

    const unsigned int threadTag = static_cast<unsigned int>(threadIdx.x) << (32 - BUCKET_WARP_LOG_SIZE);
    const int warpBase = (threadIdx.x >> BUCKET_WARP_LOG_SIZE) * DIVISIONS;
    const int numThreads = blockDim.x * gridDim.x;

    // Initialize shared memory bucket counters
    for (int i = threadIdx.x; i < BUCKET_BLOCK_MEMORY; i += blockDim.x) {
        s_offset[i] = 0U;
    }

    __syncthreads();

    // Process elements in a grid-stride loop
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < size;
         tid += numThreads) {
        float elem = __ldg(&input[tid]);

        int idx = DIVISIONS / 2 - 1;
        int jump = DIVISIONS / 4;
        float piv = tex1Dfetch<float>(texPivot, idx);

        // Binary search over pivot points
#pragma unroll
        while (jump >= 1) {
            idx = (elem < piv) ? (idx - jump) : (idx + jump);
            piv = tex1Dfetch<float>(texPivot, idx);
            jump >>= 1;
        }
        idx = (elem < piv) ? idx : (idx + 1);

        unsigned int local = static_cast<unsigned int>(
            addOffset(s_offset + warpBase, static_cast<unsigned int>(idx), threadTag)
        );

        indice[tid] =
            static_cast<int>((local << LOG_DIVISIONS) + static_cast<unsigned int>(idx));
    }

    __syncthreads();

    const int prefixBase = blockIdx.x * BUCKET_BLOCK_MEMORY;

    // Write bucket counts back to global memory
    for (int i = threadIdx.x; i < BUCKET_BLOCK_MEMORY; i += blockDim.x) {
        d_prefixoffsets[prefixBase + i] = s_offset[i] & 0x07FFFFFFU;
    }
}

__global__ void __launch_bounds__(DIVISIONS, 4)
bucketprefixoffset(unsigned int * __restrict__ d_prefixoffsets,
                   unsigned int * __restrict__ d_offsets,
                   int blocks) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blocks * BUCKET_BLOCK_MEMORY;
    unsigned int sum = 0U;

    // Stride by DIVISIONS (bucket count); this mapping is preserved
    for (int i = tid; i < size; i += DIVISIONS) {
        unsigned int x = d_prefixoffsets[i];
        d_prefixoffsets[i] = sum;
        sum += x;
    }

    d_offsets[tid] = sum;
}

__global__ void __launch_bounds__(BUCKET_THREAD_N, 2)
bucketsort(float * __restrict__ input,
           int * __restrict__ indice,
           float * __restrict__ output,
           int size,
           unsigned int * __restrict__ d_prefixoffsets,
           unsigned int * __restrict__ l_offsets) {
    extern __shared__ unsigned int s_offset_dyn[];
    volatile unsigned int *s_offset = s_offset_dyn;

    const int prefixBase = blockIdx.x * BUCKET_BLOCK_MEMORY;
    const int warpBase = (threadIdx.x >> BUCKET_WARP_LOG_SIZE) * DIVISIONS;
    const int numThreads = blockDim.x * gridDim.x;

    // Preload global offsets into shared memory
    for (int i = threadIdx.x; i < BUCKET_BLOCK_MEMORY; i += blockDim.x) {
        unsigned int localBucket = static_cast<unsigned int>(i & (DIVISIONS - 1));
        s_offset[i] =
            l_offsets[localBucket] + d_prefixoffsets[prefixBase + i];
    }

    __syncthreads();

    // Scatter elements based on computed bucket indices
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < size;
         tid += numThreads) {

        float elem = __ldg(&input[tid]);
        int id = indice[tid];

        int bucket = id & (DIVISIONS - 1);
        int localIndex = id >> LOG_DIVISIONS;
        unsigned int base = s_offset[warpBase + bucket];

        output[base + static_cast<unsigned int>(localIndex)] = elem;

        // 'test' preserved to keep original behavior (even if unused)
        int test = static_cast<int>(base) + localIndex;
        (void)test;
    }
}

#endif
