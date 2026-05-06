#ifndef _VLC_SM64HUFF_KERNEL_H_
#define _VLC_SM64HUFF_KERNEL_H_

#include "pabio_kernels_v2.cu"
#include "parameters.h"

#ifdef SMATOMICS

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

    unsigned long long cw64 = 0ull;
    unsigned int val32, codewordlen = 0u;
    unsigned char tmpbyte, tmpcwlen;
    unsigned int tmpcw32;

    extern __shared__ unsigned int sm[];
    __shared__ unsigned int kcmax;

#ifdef CACHECWLUT
    unsigned int * __restrict__ codewords    = (unsigned int *)sm;
    unsigned int * __restrict__ codewordlens = (unsigned int *)(sm + NUM_SYMBOLS);
    unsigned int * __restrict__ as           = (unsigned int *)(sm + 2 * NUM_SYMBOLS);

    // Load the codewords and the original data
    // Assume NUM_SYMBOLS is multiple of blockDim.x or large enough; use strided load
    for (unsigned int idx = k; idx < NUM_SYMBOLS; idx += blockDim.x) {
        codewords[idx]    = gm_codewords[idx];
        codewordlens[idx] = gm_codewordlens[idx];
    }

    // Ensure LUTs are fully loaded before using
    __syncthreads();

    val32 = __ldg(&data[kn]);

    // Process 4 bytes of val32, unrolled for ILP
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        tmpbyte  = static_cast<unsigned char>(val32 >> ((3 - i) * 8));
        tmpcw32  = codewords[tmpbyte];
        tmpcwlen = static_cast<unsigned char>(codewordlens[tmpbyte]);
        cw64     = (cw64 << tmpcwlen) | static_cast<unsigned long long>(tmpcw32);
        codewordlen += static_cast<unsigned int>(tmpcwlen);
    }
#else
    unsigned int * __restrict__ as = (unsigned int *)sm;

    val32 = __ldg(&data[kn]);

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        tmpbyte  = static_cast<unsigned char>(val32 >> ((3 - i) * 8));
        tmpcw32  = __ldg(&gm_codewords[tmpbyte]);
        tmpcwlen = static_cast<unsigned char>(__ldg(&gm_codewordlens[tmpbyte]));
        cw64     = (cw64 << tmpcwlen) | static_cast<unsigned long long>(tmpcw32);
        codewordlen += static_cast<unsigned int>(tmpcwlen);
    }
#endif

    // Store per-thread codeword bit-length
    as[k] = codewordlen;
    __syncthreads();

    // Parallel exclusive scan (Blelloch) over as[], inplace
    unsigned int offset = 1u;

    // Up-sweep
    for (unsigned int d = blockDim.x >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (k < d) {
            const unsigned int ai = offset * ((k << 1) + 1) - 1u;
            const unsigned int bi = offset * ((k << 1) + 2) - 1u;
            as[bi] += as[ai];
        }
        offset <<= 1;
    }

    // Clear last element to convert to exclusive scan
    if (k == 0) {
        as[blockDim.x - 1] = 0u;
    }

    // Down-sweep
    for (unsigned int d = 1u; d < blockDim.x; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (k < d) {
            const unsigned int ai = offset * ((k << 1) + 1) - 1u;
            const unsigned int bi = offset * ((k << 1) + 2) - 1u;
            const unsigned int t  = as[ai];
            as[ai] = as[bi];
            as[bi] += t;
        }
    }
    __syncthreads();

    // Compute block output size and max index written
    if (k == blockDim.x - 1) {
        const unsigned int totalBits = as[k] + codewordlen;
        outidx[blockIdx.x] = totalBits;
        kcmax = totalBits >> 5; // divide by 32
    }

    // Per-thread starting word and bit offset
    kc       = as[k] >> 5;
    startbit = as[k] & 31u;

    // Reuse shared memory buffer as[k] as temporary 0-initialized bitstream
    as[k] = 0u;
    __syncthreads();

    // PART 1: fill remainder of first 32-bit word
    wrbits = (codewordlen > (32u - startbit)) ? (32u - startbit) : codewordlen;
    if (wrbits) {
        tmpcw32 = static_cast<unsigned int>(cw64 >> (codewordlen - wrbits));
        atomicOr(&as[kc], tmpcw32 << (32u - startbit - wrbits));
        codewordlen -= wrbits;
    }

    // PART 2: next full/partial 32-bit word
    if (codewordlen) {
        wrbits  = (codewordlen > 32u) ? 32u : codewordlen;
        tmpcw32 = static_cast<unsigned int>(cw64 >> (codewordlen - wrbits)) &
                  ((1u << wrbits) - 1u);
        atomicOr(&as[kc + 1u], tmpcw32 << (32u - wrbits));
        codewordlen -= wrbits;
    }

    // PART 3: any remaining bits
    if (codewordlen) {
        tmpcw32 = static_cast<unsigned int>(cw64 & ((1ull << codewordlen) - 1ull));
        atomicOr(&as[kc + 2u], tmpcw32 << (32u - codewordlen));
    }

    __syncthreads();

    // Coalesced write of the resulting 32-bit words to global memory
    if (k <= kcmax) {
        out[kn] = as[k];
    }
}
//////////////////////////////////////////////////////////////////////////////
#endif

#endif
