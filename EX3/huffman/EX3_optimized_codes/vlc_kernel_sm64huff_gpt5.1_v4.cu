#ifndef _VLC_SM64HUFF_KERNEL_H_
#define _VLC_SM64HUFF_KERNEL_H_

#include "pabio_kernels_v2.cu"
#include "parameters.h"

#ifdef SMATOMICS

// Assume NUM_SYMBOLS == 256, safe to use 8-bit indices for LUT
// Use __restrict__ and __ldg for better caching of global loads
__global__ void
vlc_encode_kernel_sm64huff(unsigned int *__restrict__ data,
                           const unsigned int *__restrict__ gm_codewords,
                           const unsigned int *__restrict__ gm_codewordlens,
#ifdef TESTING
                           unsigned int *cw32, unsigned int *cw32len,
                           unsigned int *cw32idx,
#endif
                           unsigned int *__restrict__ out,
                           unsigned int *__restrict__ outidx) {

    const unsigned int kn = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int k  = threadIdx.x;

    unsigned int kc, startbit, wrbits;

    unsigned long long cw64 = 0ULL;
    unsigned int val32, codewordlen = 0;
    unsigned char tmpbyte, tmpcwlen;
    unsigned int tmpcw32;

    extern __shared__ unsigned int sm[];
    __shared__ unsigned int kcmax;

#ifdef CACHECWLUT
    // Layout: [NUM_SYMBOLS codewords][NUM_SYMBOLS codewordlens][blockDim.x as[]]
    unsigned int *codewords    = (unsigned int *)sm;
    unsigned int *codewordlens = (unsigned int *)(sm + NUM_SYMBOLS);
    unsigned int *as           = (unsigned int *)(sm + 2 * NUM_SYMBOLS);

    // Coalesced load of LUT per block (only first NUM_SYMBOLS threads participate)
    if (k < NUM_SYMBOLS) {
        codewords[k]    = __ldg(&gm_codewords[k]);
        codewordlens[k] = __ldg(&gm_codewordlens[k]);
    }

    // Coalesced load of input data
    val32 = __ldg(&data[kn]);

    __syncthreads();

    // Unroll fixed 4-byte loop for better ILP and to reduce loop overhead
#pragma unroll
    for (unsigned int i = 0; i < 4; i++) {
        tmpbyte  = static_cast<unsigned char>(val32 >> ((3 - i) * 8));
        tmpcw32  = codewords[tmpbyte];
        tmpcwlen = static_cast<unsigned char>(codewordlens[tmpbyte]);
        cw64     = (cw64 << tmpcwlen) | static_cast<unsigned long long>(tmpcw32);
        codewordlen += tmpcwlen;
    }
#else
    unsigned int *as = (unsigned int *)sm;

    val32 = __ldg(&data[kn]);

#pragma unroll
    for (unsigned int i = 0; i < 4; i++) {
        tmpbyte  = static_cast<unsigned char>(val32 >> ((3 - i) * 8));
        tmpcw32  = __ldg(&gm_codewords[tmpbyte]);
        tmpcwlen = static_cast<unsigned char>(__ldg(&gm_codewordlens[tmpbyte]));
        cw64     = (cw64 << tmpcwlen) | static_cast<unsigned long long>(tmpcw32);
        codewordlen += tmpcwlen;
    }
#endif

    // Store codeword length (in bits) per element
    as[k] = codewordlen;
    __syncthreads();

    // Block-wide exclusive prefix sum of codeword lengths (Blelloch scan)
    // Use 32-bit indices for safety; tree indices fit in 16-bit, but avoid char overflow
    unsigned int offset = 1;

    // Up-sweep (reduce) phase
    for (unsigned int d = (blockDim.x >> 1); d > 0; d >>= 1) {
        __syncthreads();
        if (k < d) {
            unsigned int ai = offset * ((k << 1) + 1) - 1;
            unsigned int bi = offset * ((k << 1) + 2) - 1;
            as[bi] += as[ai];
        }
        offset <<= 1;
    }

    // Clear the last element to make it exclusive scan
    if (k == 0) {
        as[blockDim.x - 1] = 0U;
    }

    // Down-sweep phase
    for (unsigned int d = 1; d < blockDim.x; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (k < d) {
            unsigned int ai = offset * ((k << 1) + 1) - 1;
            unsigned int bi = offset * ((k << 1) + 2) - 1;
            unsigned int t  = as[ai];
            as[ai]          = as[bi];
            as[bi]         += t;
        }
    }
    __syncthreads();

    // Last thread in block computes total bits and number of 32-bit words
    if (k == blockDim.x - 1) {
        const unsigned int total_bits = as[k] + codewordlen;
        outidx[blockIdx.x]            = total_bits;
        kcmax                         = total_bits >> 5; // total_bits / 32
    }

    // Determine starting word and bit position
    kc       = as[k] >> 5;   // /32
    startbit = as[k] & 31U;  // %32

    // Clear shared memory used for packed output
    as[k] = 0U;
    __syncthreads();

    // PART 1: fill remaining bits in starting 32-bit word
    wrbits  = (codewordlen > (32U - startbit)) ? (32U - startbit) : codewordlen;
    tmpcw32 = static_cast<unsigned int>(cw64 >> (codewordlen - wrbits));
    atomicOr(&as[kc],
             tmpcw32 << (32U - startbit - wrbits));
    codewordlen -= wrbits;

    // PART 2: next full/partial 32-bit word
    if (codewordlen) {
        wrbits  = (codewordlen > 32U) ? 32U : codewordlen;
        tmpcw32 = static_cast<unsigned int>(cw64 >> (codewordlen - wrbits)) &
                  ((wrbits == 32U) ? 0xFFFFFFFFU : ((1U << wrbits) - 1U));
        atomicOr(&as[kc + 1],
                 tmpcw32 << (32U - wrbits));
        codewordlen -= wrbits;
    }

    // PART 3: remaining bits (at most 32)
    if (codewordlen) {
        tmpcw32 = static_cast<unsigned int>(cw64 &
                                            ((codewordlen == 32U)
                                                 ? 0xFFFFFFFFULL
                                                 : ((1ULL << codewordlen) - 1ULL)));
        atomicOr(&as[kc + 2],
                 tmpcw32 << (32U - codewordlen));
    }

    __syncthreads();

    // Coalesced writeout of packed 32-bit words
    if (k <= kcmax) {
        out[kn] = as[k];
    }
}
//////////////////////////////////////////////////////////////////////////////
#endif

#endif
