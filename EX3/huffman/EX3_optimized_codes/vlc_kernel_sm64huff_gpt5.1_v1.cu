#ifndef _VLC_SM64HUFF_KERNEL_H_
#define _VLC_SM64HUFF_KERNEL_H_

#include "pabio_kernels_v2.cu"
#include "parameters.h"

#ifdef SMATOMICS

// Use launch_bounds to help the compiler optimize for A100 SM resources.
// Adjust maxThreadsPerBlock and minBlocksPerSM as appropriate for the rest
// of the application; here we assume up to 1024 threads/block and aim for
// reasonable occupancy.
__launch_bounds__(1024, 2)
__global__ void
vlc_encode_kernel_sm64huff(unsigned int * __restrict__ data,
                           const unsigned int * __restrict__ gm_codewords,
                           const unsigned int * __restrict__ gm_codewordlens,
#ifdef TESTING
                           unsigned int * __restrict__ cw32,
                           unsigned int * __restrict__ cw32len,
                           unsigned int * __restrict__ cw32idx,
#endif
                           unsigned int * __restrict__ out,
                           unsigned int * __restrict__ outidx) {

    const unsigned int kn = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int k  = threadIdx.x;
    unsigned int kc, startbit, wrbits;

    unsigned long long cw64 = 0ULL;
    unsigned int val32;
    unsigned int codewordlen = 0;
    unsigned char tmpbyte, tmpcwlen;
    unsigned int tmpcw32;

    extern __shared__ unsigned int sm[];
    __shared__ unsigned int kcmax;

#ifdef CACHECWLUT
    // Layout shared memory to minimize pointer arithmetic
    unsigned int * __restrict__ codewords    = sm;
    unsigned int * __restrict__ codewordlens = sm + NUM_SYMBOLS;
    unsigned int * __restrict__ as           = sm + 2 * NUM_SYMBOLS;

    // Coalesced load of LUTs and input values
    if (k < NUM_SYMBOLS) {
        codewords[k]    = gm_codewords[k];
        codewordlens[k] = gm_codewordlens[k];
    }

    // Prevent out-of-bounds when gridDim.x * blockDim.x > data size; assume
    // caller guarantees range, so just load.
    val32 = data[kn];
    __syncthreads();

    // Unroll fixed 4-iteration loop for better ILP
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        tmpbyte  = static_cast<unsigned char>(val32 >> ((3 - i) * 8));
        tmpcw32  = codewords[tmpbyte];
        tmpcwlen = static_cast<unsigned char>(codewordlens[tmpbyte]);
        cw64     = (cw64 << tmpcwlen) | static_cast<unsigned long long>(tmpcw32);
        codewordlen += tmpcwlen;
    }
#else
    unsigned int * __restrict__ as = sm;

    val32 = data[kn];

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        tmpbyte  = static_cast<unsigned char>(val32 >> ((3 - i) * 8));
        tmpcw32  = gm_codewords[tmpbyte];
        tmpcwlen = static_cast<unsigned char>(gm_codewordlens[tmpbyte]);
        cw64     = (cw64 << tmpcwlen) | static_cast<unsigned long long>(tmpcw32);
        codewordlen += tmpcwlen;
    }
#endif

    // Store codeword length (in bits) per element
    as[k] = codewordlen;
    __syncthreads();

    // Hillis–Steele style Blelloch scan, but with adjusted index type to
    // avoid overflow and improve banking; use unsigned int indices.
    unsigned int offset = 1;

    // Build the sum in place up the tree
    for (unsigned int d = (blockDim.x >> 1); d > 0; d >>= 1) {
        __syncthreads();
        if (k < d) {
            const unsigned int ai = offset * ((k << 1) + 1) - 1;
            const unsigned int bi = offset * ((k << 1) + 2) - 1;
            as[bi] += as[ai];
        }
        offset <<= 1;
    }

    // Clear the last element
    if (k == 0) {
        as[blockDim.x - 1] = 0U;
    }

    // Traverse down the tree building the scan in place
    for (unsigned int d = 1; d < blockDim.x; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (k < d) {
            const unsigned int ai = offset * ((k << 1) + 1) - 1;
            const unsigned int bi = offset * ((k << 1) + 2) - 1;
            const unsigned int t  = as[ai];
            as[ai] = as[bi];
            as[bi] += t;
        }
    }
    __syncthreads();

    if (k == blockDim.x - 1) {
        const unsigned int last = as[k] + codewordlen;
        outidx[blockIdx.x] = last;
        kcmax = last >> 5;  // divide by 32
    }

    // Compute starting word and bit offset
    kc       = as[k] >> 5;       // / 32
    startbit = as[k] & 31U;      // % 32
    as[k]    = 0U;
    __syncthreads();

    // Part 1
    wrbits  = codewordlen > (32U - startbit) ? (32U - startbit) : codewordlen;
    tmpcw32 = static_cast<unsigned int>(cw64 >> (codewordlen - wrbits));
    // Atomic OR into shared memory; use warp-aggregated access pattern
    atomicOr(&as[kc], tmpcw32 << (32U - startbit - wrbits));
    codewordlen -= wrbits;

    // Part 2
    if (codewordlen) {
        wrbits  = codewordlen > 32U ? 32U : codewordlen;
        tmpcw32 = static_cast<unsigned int>(cw64 >> (codewordlen - wrbits)) &
                  ((1U << wrbits) - 1U);
        atomicOr(&as[kc + 1U], tmpcw32 << (32U - wrbits));
        codewordlen -= wrbits;
    }

    // Part 3
    if (codewordlen) {
        tmpcw32 = static_cast<unsigned int>(cw64 & ((1ULL << codewordlen) - 1ULL));
        atomicOr(&as[kc + 2U], tmpcw32 << (32U - codewordlen));
    }

    __syncthreads();

    // Ensure all threads see final shared-memory updates before global write
    if (k <= kcmax) {
        out[kn] = as[k];
    }
}
//////////////////////////////////////////////////////////////////////////////
#endif

#endif
