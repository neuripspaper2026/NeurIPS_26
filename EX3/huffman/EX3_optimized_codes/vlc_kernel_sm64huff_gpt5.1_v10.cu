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

    const unsigned int kn = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int k  = threadIdx.x;

    unsigned int kc, startbit, wrbits;

    unsigned long long cw64 = 0ull;
    unsigned int val32, codewordlen = 0u;
    unsigned char tmpbyte, tmpcwlen;
    unsigned int tmpcw32;

    extern __shared__ unsigned int sm[];
    __shared__ unsigned int kcmax;

#ifdef CACHECWLUT
    // Layout: [NUM_SYMBOLS codewords][NUM_SYMBOLS codewordlens][blockDim.x as[]]
    unsigned int *codewords    = (unsigned int *)sm;
    unsigned int *codewordlens = (unsigned int *)(sm + NUM_SYMBOLS);
    unsigned int *as           = (unsigned int *)(sm + 2 * NUM_SYMBOLS);

    // Each thread within first NUM_SYMBOLS loads one symbol; others do nothing
    if (k < NUM_SYMBOLS) {
        // Use __ldg to leverage read-only cache on A100
        codewords[k]    = __ldg(&gm_codewords[k]);
        codewordlens[k] = __ldg(&gm_codewordlens[k]);
    }

    __syncthreads();

    // Load input symbol (coalesced global read)
    val32 = data[kn];

    // Unroll fixed 4-iteration loop to reduce control overhead
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        tmpbyte  = static_cast<unsigned char>(val32 >> ((3 - i) * 8));
        tmpcw32  = codewords[tmpbyte];
        tmpcwlen = static_cast<unsigned char>(codewordlens[tmpbyte]);
        cw64     = (cw64 << tmpcwlen) | static_cast<unsigned long long>(tmpcw32);
        codewordlen += tmpcwlen;
    }
#else
    unsigned int *as = (unsigned int *)sm;

    // Coalesced load of input
    val32 = data[kn];

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        tmpbyte  = static_cast<unsigned char>(val32 >> ((3 - i) * 8));
        tmpcw32  = __ldg(&gm_codewords[tmpbyte]);
        tmpcwlen = static_cast<unsigned char>(__ldg(&gm_codewordlens[tmpbyte]));
        cw64     = (cw64 << tmpcwlen) | static_cast<unsigned long long>(tmpcw32);
        codewordlen += tmpcwlen;
    }
#endif

    // Store codeword length in bits
    as[k] = codewordlen;
    __syncthreads();

    // ---------------------------------------------
    // Blelloch scan (prefix sum) over as[]
    // ---------------------------------------------

    // Build the sum in place up the tree
    unsigned int offset = 1u;

    // Use unsigned int for indices; avoid overflow from unsigned char
    for (unsigned int d = (blockDim.x >> 1); d > 0; d >>= 1) {
        __syncthreads();
        if (k < d) {
            unsigned int ai = offset * ((k << 1) + 1u) - 1u;
            unsigned int bi = offset * ((k << 1) + 2u) - 1u;
            as[bi] += as[ai];
        }
        offset <<= 1;
    }

    // Clear the last element
    if (k == 0) {
        as[blockDim.x - 1] = 0u;
    }

    // Traverse down the tree building the scan in place
    for (unsigned int d = 1u; d < blockDim.x; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (k < d) {
            unsigned int ai = offset * ((k << 1) + 1u) - 1u;
            unsigned int bi = offset * ((k << 1) + 2u) - 1u;
            unsigned int t  = as[ai];
            as[ai]          = as[bi];
            as[bi]         += t;
        }
    }
    __syncthreads();

    // Compute per-block output index and maximum word index
    if (k == blockDim.x - 1) {
        const unsigned int totalBits = as[k] + codewordlen;
        outidx[blockIdx.x] = totalBits;
        kcmax = totalBits >> 5; // divide by 32
    }

    // ---------------------------------------------
    // Write the codes into shared buffer using atomics
    // ---------------------------------------------
    kc       = as[k] >> 5;   // /32
    startbit = as[k] & 31u;  // %32
    as[k]    = 0u;
    __syncthreads();

    // Part 1
    wrbits  = codewordlen > (32u - startbit) ? (32u - startbit) : codewordlen;
    tmpcw32 = static_cast<unsigned int>(cw64 >> (codewordlen - wrbits));

    atomicOr(&as[kc], tmpcw32 << (32u - startbit - wrbits));
    codewordlen -= wrbits;

    // Part 2
    if (codewordlen) {
        wrbits  = codewordlen > 32u ? 32u : codewordlen;
        tmpcw32 = static_cast<unsigned int>(
                      (cw64 >> (codewordlen - wrbits)) & ((1ull << wrbits) - 1ull));
        atomicOr(&as[kc + 1], tmpcw32 << (32u - wrbits));
        codewordlen -= wrbits;
    }

    // Part 3
    if (codewordlen) {
        tmpcw32 = static_cast<unsigned int>(cw64 & ((1ull << codewordlen) - 1ull));
        atomicOr(&as[kc + 2], tmpcw32 << (32u - codewordlen));
    }

    __syncthreads();

    // Coalesced writeback to global memory
    if (k <= kcmax) {
        out[kn] = as[k];
    }
}
//////////////////////////////////////////////////////////////////////////////
#endif

#endif
