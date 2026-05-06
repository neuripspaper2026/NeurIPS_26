#ifndef _VLC_SM64HUFF_KERNEL_H_
#define _VLC_SM64HUFF_KERNEL_H_

#include "pabio_kernels_v2.cu"
#include "parameters.h"

#ifdef SMATOMICS

__global__ void
vlc_encode_kernel_sm64huff(unsigned int *data, const unsigned int *gm_codewords,
                           const unsigned int *gm_codewordlens,
#ifdef TESTING
                           unsigned int *cw32, unsigned int *cw32len,
                           unsigned int *cw32idx,
#endif
                           unsigned int *out, unsigned int *outidx) {

    // Use const/constexpr where possible and keep indices 32-bit
    const unsigned int kn = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int k  = threadIdx.x;

    unsigned int kc, startbit, wrbits;

    // Keep cw64 in register, avoid unnecessary type changes
    unsigned long long cw64 = 0ull;
    unsigned int val32;
    unsigned int codewordlen = 0u;
    unsigned char tmpbyte, tmpcwlen;
    unsigned int tmpcw32;

    extern __shared__ unsigned int sm[];
    __shared__ unsigned int kcmax;

#ifdef CACHECWLUT
    // Lay out shared memory carefully to keep accesses coalesced and avoid bank conflicts
    unsigned int *codewords    = sm;
    unsigned int *codewordlens = sm + NUM_SYMBOLS;
    unsigned int *as           = sm + 2 * NUM_SYMBOLS;

    // Use cached loads for LUT and data
    // Coalesced across threads in a block
    if (k < NUM_SYMBOLS) {
        codewords[k]    = __ldg(&gm_codewords[k]);
        codewordlens[k] = __ldg(&gm_codewordlens[k]);
    }

    // All threads in the block must sync before using LUT
    __syncthreads();

    // Read input data: coalesced along x dimension
    val32 = __ldg(&data[kn]);

    // Unroll the small fixed loop to reduce control overhead and improve ILP
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        tmpbyte  = static_cast<unsigned char>(val32 >> ((3 - i) * 8));
        tmpcw32  = codewords[tmpbyte];
        tmpcwlen = static_cast<unsigned char>(codewordlens[tmpbyte]);
        cw64     = (cw64 << tmpcwlen) | static_cast<unsigned long long>(tmpcw32);
        codewordlen += tmpcwlen;
    }
#else
    unsigned int *as = sm;

    val32 = __ldg(&data[kn]);

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        tmpbyte  = static_cast<unsigned char>(val32 >> ((3 - i) * 8));
        tmpcw32  = __ldg(&gm_codewords[tmpbyte]);
        tmpcwlen = static_cast<unsigned char>(__ldg(&gm_codewordlens[tmpbyte]));
        cw64     = (cw64 << tmpcwlen) | static_cast<unsigned long long>(tmpcw32);
        codewordlen += tmpcwlen;
    }
#endif

    // Store codeword length in shared memory; as[] will be reused as scan buffer
    as[k] = codewordlen;
    __syncthreads();

    // Blelloch exclusive scan over as[], with minor type fixes and avoiding
    // recalculations; blockDim.x is assumed power-of-two in original code.
    unsigned int offset = 1u;

    // Build the sum in place up the tree
    for (unsigned int d = (blockDim.x >> 1); d > 0; d >>= 1) {
        __syncthreads();
        if (k < d) {
            const unsigned int ai = offset * ((k << 1) + 1u) - 1u;
            const unsigned int bi = offset * ((k << 1) + 2u) - 1u;
            as[bi] += as[ai];
        }
        offset <<= 1;
    }

    // Clear the last element
    if (k == 0u)
        as[blockDim.x - 1u] = 0u;

    // Traverse down the tree building the scan in place
    for (unsigned int d = 1u; d < blockDim.x; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (k < d) {
            const unsigned int ai = offset * ((k << 1) + 1u) - 1u;
            const unsigned int bi = offset * ((k << 1) + 2u) - 1u;
            const unsigned int t  = as[ai];
            as[ai] = as[bi];
            as[bi] += t;
        }
    }
    __syncthreads();

    // Last thread stores block-wide bit count and kcmax
    if (k == blockDim.x - 1u) {
        const unsigned int last = as[k] + codewordlen;
        outidx[blockIdx.x] = last;
        kcmax = last >> 5;  // divide by 32
    }

    // Compute word index and starting bit for this thread
    kc       = as[k] >> 5;       // / 32
    startbit = as[k] & 31u;      // % 32

    // Reuse as[] as zeroed 32-bit buffer for packed output
    as[k] = 0u;
    __syncthreads();

    // Part 1: fill remaining bits of first 32-bit word
    wrbits = codewordlen > (32u - startbit) ? (32u - startbit) : codewordlen;
    if (wrbits) {
        tmpcw32 = static_cast<unsigned int>(cw64 >> (codewordlen - wrbits));
        // shift left in case it's shorter than available space
        atomicOr(&as[kc], tmpcw32 << (32u - startbit - wrbits));
        codewordlen -= wrbits;
    }

    // Part 2: next full or partial 32-bit word
    if (codewordlen) {
        wrbits  = codewordlen > 32u ? 32u : codewordlen;
        tmpcw32 = static_cast<unsigned int>(cw64 >> (codewordlen - wrbits)) &
                  ((wrbits == 32u) ? 0xFFFFFFFFu : ((1u << wrbits) - 1u));
        atomicOr(&as[kc + 1u], tmpcw32 << (32u - wrbits));
        codewordlen -= wrbits;
    }

    // Part 3: remainder, at most 32 bits
    if (codewordlen) {
        tmpcw32 = static_cast<unsigned int>(cw64 &
                  ((codewordlen == 32u) ? 0xFFFFFFFFu : ((1u << codewordlen) - 1u)));
        atomicOr(&as[kc + 2u], tmpcw32 << (32u - codewordlen));
    }

    __syncthreads();

    // Only threads that wrote valid 32-bit words commit to global memory.
    // Use per-block contiguous region; writes are coalesced for kn within block.
    if (k <= kcmax)
        out[kn] = as[k];
}
//////////////////////////////////////////////////////////////////////////////
#endif

#endif
