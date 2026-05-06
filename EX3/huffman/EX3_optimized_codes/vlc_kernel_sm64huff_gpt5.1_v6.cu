#ifndef _VLC_SM64HUFF_KERNEL_H_
#define _VLC_SM64HUFF_KERNEL_H_

#include "pabio_kernels_v2.cu"
#include "parameters.h"

#ifdef SMATOMICS

// Use warp-synchronous prefix sum on A100 for better performance.
// Assumes blockDim.x is a power-of-two and <= 1024.
__device__ __forceinline__ unsigned int warp_prefix_sum_inclusive(unsigned int val, unsigned int lane) {
    // Tree-based inclusive scan within a warp using shuffle
    for (int offset = 1; offset < 32; offset <<= 1) {
        unsigned int n = __shfl_up_sync(0xffffffffu, val, offset);
        if (lane >= (unsigned)offset) {
            val += n;
        }
    }
    return val;
}

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
    unsigned long long cw64 = 0ULL;
    unsigned int val32, codewordlen = 0U;
    unsigned char tmpbyte, tmpcwlen;
    unsigned int tmpcw32;

    extern __shared__ unsigned int sm[];
    __shared__ unsigned int kcmax;

#ifdef CACHECWLUT
    // Layout: [codewords | codewordlens | as]
    unsigned int *codewords    = (unsigned int *)sm;
    unsigned int *codewordlens = (unsigned int *)(sm + NUM_SYMBOLS);
    unsigned int *as           = (unsigned int *)(sm + 2 * NUM_SYMBOLS);

    // Coalesced load of LUT
    if (k < NUM_SYMBOLS) {
        codewords[k]    = gm_codewords[k];
        codewordlens[k] = gm_codewordlens[k];
    }
    __syncthreads();

    // Load the original data (coalesced across threads in block)
    val32 = data[kn];

    // Build up to 32-bit (max) Huffman codeword in 64-bit accumulator
#pragma unroll
    for (unsigned int i = 0; i < 4; i++) {
        tmpbyte  = (unsigned char)(val32 >> ((3 - i) * 8));
        tmpcw32  = codewords[tmpbyte];
        tmpcwlen = (unsigned char)codewordlens[tmpbyte];
        cw64     = (cw64 << tmpcwlen) | (unsigned long long)tmpcw32;
        codewordlen += (unsigned int)tmpcwlen;
    }
#else
    // Layout: [as]
    unsigned int *as = (unsigned int *)sm;

    val32 = data[kn];

#pragma unroll
    for (unsigned int i = 0; i < 4; i++) {
        tmpbyte  = (unsigned char)(val32 >> ((3 - i) * 8));
        tmpcw32  = gm_codewords[tmpbyte];
        tmpcwlen = (unsigned char)gm_codewordlens[tmpbyte];
        cw64     = (cw64 << tmpcwlen) | (unsigned long long)tmpcw32;
        codewordlen += (unsigned int)tmpcwlen;
    }
#endif

    // Store per-thread codeword length in bits
    as[k] = codewordlen;
    __syncthreads();

    // ----------------- Optimized prefix sum (scan) -----------------
    // Use warp-level scan + shared-memory reduction instead of
    // classical Blelloch scan, which reduces synchronizations.
    const unsigned int lane   = threadIdx.x & 31;
    const unsigned int warpId = threadIdx.x >> 5;
    const unsigned int numWarps = (blockDim.x + 31) >> 5;

    // Step 1: intra-warp inclusive scan
    unsigned int val = as[k];
    val = warp_prefix_sum_inclusive(val, lane);
    as[k] = val;

    __syncthreads();

    // Step 2: each warp's last lane writes its total to shared memory
    if (lane == 31) {
        sm[numWarps + warpId] = val;
    }
    __syncthreads();

    // Step 3: first warp scans warp sums to get prefix for each warp
    if (warpId == 0) {
        unsigned int warpVal = 0U;
        if (lane < numWarps) {
            warpVal = sm[numWarps + lane];
            warpVal = warp_prefix_sum_inclusive(warpVal, lane);
            sm[numWarps + lane] = warpVal;
        }
    }
    __syncthreads();

    // Step 4: add warp prefix (exclusive) to each thread's intra-warp scan
    if (warpId > 0) {
        unsigned int warpPrefix = sm[numWarps + warpId - 1];
        val += warpPrefix;
    }
    // Convert inclusive scan to exclusive by subtracting own length
    as[k] = val - codewordlen;
    __syncthreads();
    // ----------------- End optimized prefix sum -----------------

    if (k == blockDim.x - 1) {
        const unsigned int totalBits = as[k] + codewordlen;
        outidx[blockIdx.x] = totalBits;
        kcmax = totalBits >> 5; // totalBits / 32
    }

    // Compute starting word and bit offset for this thread
    kc       = as[k] >> 5;      // /32
    startbit = as[k] & 31;      // %32
    as[k]    = 0U;
    __syncthreads();

    // ----------------- Write the codes using shared atomics -----------------
    // Part 1
    wrbits   = codewordlen > (32U - startbit) ? (32U - startbit) : codewordlen;
    tmpcw32  = (unsigned int)(cw64 >> (codewordlen - wrbits));

    // No branch on wrbits == 32 to avoid divergence
    atomicOr(&as[kc], tmpcw32 << (32U - startbit - wrbits));
    codewordlen -= wrbits;

    // Part 2
    if (codewordlen) {
        wrbits  = codewordlen > 32U ? 32U : codewordlen;
        tmpcw32 = (unsigned int)(cw64 >> (codewordlen - wrbits)) &
                  ((wrbits == 32U) ? 0xffffffffu : ((1U << wrbits) - 1U));
        atomicOr(&as[kc + 1U], tmpcw32 << (32U - wrbits));
        codewordlen -= wrbits;
    }

    // Part 3
    if (codewordlen) {
        tmpcw32 = (unsigned int)(cw64 & ((codewordlen == 32U) ? 0xffffffffu : ((1U << codewordlen) - 1U)));
        atomicOr(&as[kc + 2U], tmpcw32 << (32U - codewordlen));
    }

    __syncthreads();

    // Coalesced write of encoded 32-bit words to global memory
    if (k <= kcmax) {
        out[kn] = as[k];
    }
}

//////////////////////////////////////////////////////////////////////////////
#endif

#endif
