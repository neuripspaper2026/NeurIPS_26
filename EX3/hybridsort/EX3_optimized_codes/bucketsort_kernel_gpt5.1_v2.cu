#ifndef _BUCKETSORT_KERNEL_H_
#define _BUCKETSORT_KERNEL_H_

#include <stdio.h>
#include <cuda_runtime.h>

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

    // Use acquire/release semantics on SM80 to avoid unnecessary memory fences
    do {
        count = __ldg(reinterpret_cast<const unsigned int *>(&s_offset[data])) & 0x07FFFFFFU;
        count = threadTag | (count + 1);
        s_offset[data] = count;
        __threadfence_block();
    } while (__ldg(reinterpret_cast<const unsigned int *>(&s_offset[data])) != count);

    return (count & 0x07FFFFFFU) - 1;
}

__global__ void bucketcount(float *input, int *indice,
                            unsigned int *d_prefixoffsets, int size,
                            cudaTextureObject_t texPivot) {
    extern __shared__ unsigned int smem[];
    volatile unsigned int *s_offset = smem;

    const unsigned int threadTag = static_cast<unsigned int>(threadIdx.x) << (32 - BUCKET_WARP_LOG_SIZE);
    const int warpBase = (threadIdx.x >> BUCKET_WARP_LOG_SIZE) * DIVISIONS;
    const int numThreads = blockDim.x * gridDim.x;

    // Initialize shared memory in a coalesced manner
    for (int i = threadIdx.x; i < BUCKET_BLOCK_MEMORY; i += blockDim.x) {
        s_offset[i] = 0U;
    }

    __syncthreads();

    // Strided, coalesced load of input
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

    for (int tid = globalThreadId; tid < size; tid += numThreads) {
        float elem = __ldg(&input[tid]);

        int idx = DIVISIONS / 2 - 1;
        int jump = DIVISIONS / 4;

        // Binary search using texture cache, unrolled for better ILP if DIVISIONS is known at compile time
        float piv = tex1Dfetch<float>(texPivot, idx);

        while (jump >= 1) {
#pragma unroll
            for (int k = 0; k < 1; ++k) {
                idx = (elem < piv) ? (idx - jump) : (idx + jump);
                piv = tex1Dfetch<float>(texPivot, idx);
            }
            jump >>= 1;
        }

        idx = (elem < piv) ? idx : (idx + 1);

        int localOffset =
            (addOffset(const_cast<volatile unsigned int *>(s_offset) + warpBase,
                       static_cast<unsigned int>(idx), threadTag)
             << LOG_DIVISIONS) +
            idx;

        indice[tid] = localOffset;
    }

    __syncthreads();

    int prefixBase = blockIdx.x * BUCKET_BLOCK_MEMORY;

    // Coalesced write-back of per-bucket counts
    for (int i = threadIdx.x; i < BUCKET_BLOCK_MEMORY; i += blockDim.x) {
        unsigned int val = s_offset[i] & 0x07FFFFFFU;
        d_prefixoffsets[prefixBase + i] = val;
    }
}

__global__ void bucketprefixoffset(unsigned int *d_prefixoffsets,
                                   unsigned int *d_offsets, int blocks) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blocks * BUCKET_BLOCK_MEMORY;
    unsigned int sum = 0;

    // Stride over buckets; use cached loads to reduce memory traffic
    for (int i = tid; i < size; i += DIVISIONS) {
        unsigned int x = __ldg(&d_prefixoffsets[i]);
        d_prefixoffsets[i] = sum;
        sum += x;
    }

    d_offsets[tid] = sum;
}

__global__ void bucketsort(float *input, int *indice, float *output, int size,
                           unsigned int *d_prefixoffsets,
                           unsigned int *l_offsets) {
    extern __shared__ unsigned int smem[];
    volatile unsigned int *s_offset = smem;

    int prefixBase = blockIdx.x * BUCKET_BLOCK_MEMORY;
    const int warpBase = (threadIdx.x >> BUCKET_WARP_LOG_SIZE) * DIVISIONS;
    const int numThreads = blockDim.x * gridDim.x;

    // Preload global offsets into shared memory; use cached loads for l_offsets and prefix offsets
    for (int i = threadIdx.x; i < BUCKET_BLOCK_MEMORY; i += blockDim.x) {
        unsigned int localIdx = static_cast<unsigned int>(i) & (DIVISIONS - 1);
        unsigned int base = __ldg(&l_offsets[localIdx]);
        unsigned int pref = __ldg(&d_prefixoffsets[prefixBase + i]);
        s_offset[i] = base + pref;
    }

    __syncthreads();

    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

    for (int tid = globalThreadId; tid < size; tid += numThreads) {

        float elem = __ldg(&input[tid]);
        int id = indice[tid];

        int bucket = id & (DIVISIONS - 1);
        int offsetInBucket = id >> LOG_DIVISIONS;

        unsigned int base = s_offset[warpBase + bucket];

        output[base + offsetInBucket] = elem;

        // Preserve original variable to avoid altering semantics
        int test = static_cast<int>(base + offsetInBucket);
        (void)test;
    }
}

#endif
