#ifndef _VLC_SM64HUFF_KERNEL_H_
#define _VLC_SM64HUFF_KERNEL_H_

#include "pabio_kernels_v2.cu"
#include "parameters.h"

#ifdef SMATOMICS

// Use 32-bit indices for shared-memory operations to avoid overflow/truncation
// and enable better compiler optimization on A100.
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

    // keep as 64-bit accumulator
    unsigned long long cw64 = 0ULL;
    unsigned int val32;
    unsigned int codewordlen = 0U;
    unsigned char tmpbyte, tmpcwlen;
    unsigned int tmpcw32;

    extern __shared__ unsigned int sm[];
    __shared__ unsigned int kcmax;

#ifdef CACHECWLUT
    // Layout: [codewords | codewordlens | as]
    unsigned int * __restrict__ codewords    = sm;
    unsigned int * __restrict__ codewordlens = sm + NUM_SYMBOLS;
    unsigned int * __restrict__ as           = sm + 2 * NUM_SYMBOLS;

    // Coalesced load of LUTs and input data
    if (k < NUM_SYMBOLS) {
        codewords[k]    = gm_codewords[k];
        codewordlens[k] = gm_codewordlens[k];
    }
    val32 = data[kn];
    __syncthreads();

    // Build 64-bit codeword locally (no shared memory traffic)
#pragma unroll
    for (int i = 0; i < 4; i++) {
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
    for (int i = 0; i < 4; i++) {
        tmpbyte  = static_cast<unsigned char>(val32 >> ((3 - i) * 8));
        tmpcw32  = gm_codewords[tmpbyte];
        tmpcwlen = static_cast<unsigned char>(gm_codewordlens[tmpbyte]);
        cw64     = (cw64 << tmpcwlen) | static_cast<unsigned long long>(tmpcw32);
        codewordlen += tmpcwlen;
    }
#endif

#ifdef TESTING
    // Preserve original testing outputs (if used elsewhere)
    if (cw32 && cw32len && cw32idx) {
        cw32[kn]    = static_cast<unsigned int>(cw64 & 0xFFFFFFFFu);
        cw32len[kn] = codewordlen;
        cw32idx[kn] = kn;
    }
#endif

    // Store codeword lengths in shared memory
    as[k] = codewordlen;
    __syncthreads();

    // In-place Blelloch scan over as[] (codeword lengths in bits)
    unsigned int offset = 1;

    // Up-sweep / reduce phase
    for (unsigned int d = (blockDim.x >> 1); d > 0; d >>= 1) {
        __syncthreads();
        if (k < d) {
            const unsigned int ai = offset * ((k << 1) + 1) - 1;
            const unsigned int bi = offset * ((k << 1) + 2) - 1;
            as[bi] += as[ai];
        }
        offset <<= 1;
    }

    // Clear last element for down-sweep
    if (k == 0) {
        as[blockDim.x - 1] = 0U;
    }

    // Down-sweep phase
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

    // Last thread in block writes total bit-length and number of full 32-bit words
    if (k == blockDim.x - 1) {
        const unsigned int total_bits = as[k] + codewordlen;
        outidx[blockIdx.x] = total_bits;
        kcmax = total_bits >> 5;  // divide by 32
    }

    // Compute position in 32-bit word stream
    kc       = as[k] >> 5;        // /32
    startbit = as[k] & 31U;       // %32
    as[k]    = 0U;
    __syncthreads();

    // Part 1: fill from first partial word
    wrbits = (codewordlen > (32U - startbit)) ? (32U - startbit) : codewordlen;
    tmpcw32 = static_cast<unsigned int>(cw64 >> (codewordlen - wrbits));
    atomicOr(&as[kc], tmpcw32 << (32U - startbit - wrbits));
    codewordlen -= wrbits;

    // Part 2: possible middle word
    if (codewordlen) {
        wrbits  = (codewordlen > 32U) ? 32U : codewordlen;
        tmpcw32 = static_cast<unsigned int>(
                      (cw64 >> (codewordlen - wrbits)) &
                      ((wrbits == 32U) ? 0xFFFFFFFFu : ((1u << wrbits) - 1u)));
        atomicOr(&as[kc + 1], tmpcw32 << (32U - wrbits));
        codewordlen -= wrbits;
    }

    // Part 3: possible last partial word
    if (codewordlen) {
        tmpcw32 = static_cast<unsigned int>(
                      cw64 & ((codewordlen == 32U)
                                  ? 0xFFFFFFFFu
                                  : ((1u << codewordlen) - 1u)));
        atomicOr(&as[kc + 2], tmpcw32 << (32U - codewordlen));
    }

    __syncthreads();

    // Coalesced write of packed output
    if (k <= kcmax) {
        out[kn] = as[k];
    }
}
//////////////////////////////////////////////////////////////////////////////
#endif

#endif
