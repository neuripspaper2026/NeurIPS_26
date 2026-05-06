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

// Use read-only cache for input where beneficial
#ifndef __ldg
#define __ldg(p) (*(p))
#endif

// Keep function signature and name identical; add inlining and restrict hints
__device__ __forceinline__ int addOffset(volatile unsigned int *s_offset,
                                         unsigned int data,
                                         unsigned int threadTag) {
    unsigned int count;
#pragma unroll 1
    do {
        // Mask once and then compose new count
        count = s_offset[data] & 0x07FFFFFFU;
        count = threadTag | (count + 1);
        // Volatile store is required for concurrency semantics
        s_offset[data] = count;
        // Loop until this thread's tag is observed (lock-free per-bucket incr)
    } while (s_offset[data] != count);

    return (count & 0x07FFFFFFU) - 1;
}

__global__ void bucketcount(float *input, int *indice,
                            unsigned int *d_prefixoffsets, int size,
                            cudaTextureObject_t texPivot) {
    // Align shared memory to 16 bytes to reduce bank conflicts on A100
    __shared__ __align__(16) unsigned int s_offset[BUCKET_BLOCK_MEMORY];

    const unsigned int threadTag =
        static_cast<unsigned int>(threadIdx.x) << (32 - BUCKET_WARP_LOG_SIZE);
    const int warpBase = (threadIdx.x >> BUCKET_WARP_LOG_SIZE) * DIVISIONS;
    const int numThreads = blockDim.x * gridDim.x;

    // Initialize shared memory with strided loop to keep accesses coalesced
#pragma unroll
    for (int i = threadIdx.x; i < BUCKET_BLOCK_MEMORY; i += blockDim.x) {
        s_offset[i] = 0u;
    }

    __syncthreads();

    // Main processing loop: grid-stride to fully utilize GPU
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < size;
         tid += numThreads) {
        // Use read-only cache for input on A100
        float elem = __ldg(&input[tid]);

        // Binary search over pivots using texture object
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

        // Per-warp bucket offset calculation using shared memory
        unsigned int local =
            static_cast<unsigned int>(addOffset(s_offset + warpBase,
                                                static_cast<unsigned int>(idx),
                                                threadTag));

        // Encode index: (offset << LOG_DIVISIONS) + bucket id
        indice[tid] =
            (static_cast<int>(local) << LOG_DIVISIONS) + idx;
    }

    __syncthreads();

    const int prefixBase = blockIdx.x * BUCKET_BLOCK_MEMORY;

    // Write per-block prefix offsets back to global memory
#pragma unroll
    for (int i = threadIdx.x; i < BUCKET_BLOCK_MEMORY; i += blockDim.x) {
        d_prefixoffsets[prefixBase + i] = s_offset[i] & 0x07FFFFFFU;
    }
}

__global__ void bucketprefixoffset(unsigned int *d_prefixoffsets,
                                   unsigned int *d_offsets, int blocks) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blocks * BUCKET_BLOCK_MEMORY;

    unsigned int sum = 0u;

    // Stride by DIVISIONS as in original code; use unsigned to avoid UB
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
    // Align shared memory and keep as volatile for synchronization semantics
    volatile __shared__ __align__(16) unsigned int s_offset[BUCKET_BLOCK_MEMORY];

    const int prefixBase = blockIdx.x * BUCKET_BLOCK_MEMORY;
    const int warpBase = (threadIdx.x >> BUCKET_WARP_LOG_SIZE) * DIVISIONS;
    const int numThreads = blockDim.x * gridDim.x;

    // Precompute shared offsets: local + global prefix
#pragma unroll
    for (int i = threadIdx.x; i < BUCKET_BLOCK_MEMORY; i += blockDim.x) {
        unsigned int localBase = l_offsets[i & (DIVISIONS - 1)];
        unsigned int globalBase = d_prefixoffsets[prefixBase + i];
        s_offset[i] = localBase + globalBase;
    }

    __syncthreads();

    // Scatter elements into final position; grid-stride for full utilization
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < size;
         tid += numThreads) {
        float elem = __ldg(&input[tid]);
        int id = indice[tid];

        // Decode bucket and intra-bucket offset
        int bucket = id & (DIVISIONS - 1);
        int intra = id >> LOG_DIVISIONS;

        unsigned int base = s_offset[warpBase + bucket];
        output[base + static_cast<unsigned int>(intra)] = elem;

        // Preserve original variable (even if unused) for compatibility
        int test = static_cast<int>(base + static_cast<unsigned int>(intra));
        (void)test;
    }
}

#endif
