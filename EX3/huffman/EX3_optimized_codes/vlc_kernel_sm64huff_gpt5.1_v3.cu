#ifndef _VLC_SM64HUFF_KERNEL_H_
#define _VLC_SM64HUFF_KERNEL_H_

#include "pabio_kernels_v2.cu"
#include "parameters.h"

#ifdef SMATOMICS

// Helper: warp-synchronous inclusive scan of 32-bit values.
// Assumes full warp or masked participation via 'active' mask.
__device__ __forceinline__ unsigned int warpExclusiveScan(unsigned int val,
                                                          unsigned int lane,
                                                          unsigned int active) {
    // Inclusive scan
    for (int offset = 1; offset < 32; offset <<= 1) {
        unsigned int n = __shfl_up_sync(active, val, offset);
        if (lane >= (unsigned)offset) val += n;
    }
    // Convert to exclusive
    return val - ((active & (1u << lane)) ? __shfl_sync(active, val, lane) : 0);
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
    const unsigned int lane = k & 31;
    const unsigned int warpId = k >> 5;
    const unsigned int nWarps = (blockDim.x + 31) >> 5;

    unsigned int kc, startbit, wrbits;

    unsigned long long cw64 = 0ULL;
    unsigned int val32, codewordlen = 0;
    unsigned char tmpbyte, tmpcwlen;
    unsigned int tmpcw32;

    extern __shared__ unsigned int sm[];
    __shared__ unsigned int kcmax;

#ifdef CACHECWLUT
    unsigned int *codewords    = (unsigned int *)sm;
    unsigned int *codewordlens = (unsigned int *)(sm + NUM_SYMBOLS);
    unsigned int *as           = (unsigned int *)(sm + 2 * NUM_SYMBOLS);

    // Load LUT (assumes NUM_SYMBOLS >= blockDim.x and aligned)
    if (k < NUM_SYMBOLS) {
        codewords[k]    = gm_codewords[k];
        codewordlens[k] = gm_codewordlens[k];
    }
    __syncthreads();

    // Load input coalesced
    val32 = data[kn];

    // Unroll byte processing for better ILP
#pragma unroll
    for (int i = 0; i < 4; i++) {
        tmpbyte  = (unsigned char)(val32 >> ((3 - i) * 8));
        tmpcw32  = codewords[tmpbyte];
        tmpcwlen = (unsigned char)codewordlens[tmpbyte];
        cw64     = (cw64 << tmpcwlen) | (unsigned long long)tmpcw32;
        codewordlen += (unsigned int)tmpcwlen;
    }
#else
    unsigned int *as = (unsigned int *)sm;

    val32 = data[kn];

#pragma unroll
    for (int i = 0; i < 4; i++) {
        tmpbyte  = (unsigned char)(val32 >> ((3 - i) * 8));
        tmpcw32  = gm_codewords[tmpbyte];
        tmpcwlen = (unsigned char)gm_codewordlens[tmpbyte];
        cw64     = (cw64 << tmpcwlen) | (unsigned long long)tmpcw32;
        codewordlen += (unsigned int)tmpcwlen;
    }
#endif

    // Store local codeword length
    as[k] = codewordlen;
    __syncthreads();

    // ---------------- WARP-OPTIMIZED EXCLUSIVE SCAN -----------------
    // Step 1: per-warp exclusive scan
    unsigned int val = as[k];
    unsigned int active = __ballot_sync(0xffffffffu, k < blockDim.x);
    unsigned int ex = warpExclusiveScan(val, lane, active);
    as[k] = ex; // temporary store per-thread exclusive partial

    // Step 2: collect warp sums into first lane of each warp
    __shared__ unsigned int warpSums[1024 / 32];  // support up to 1024 threads/block
    if (lane == 31 || (k + 1 == blockDim.x)) {
        // last active lane in warp writes warp sum (inclusive total)
        unsigned int warpTotal = ex + val;
        warpSums[warpId] = warpTotal;
    }
    __syncthreads();

    // Step 3: scan warp sums using first warp
    if (warpId == 0 && k < nWarps) {
        unsigned int wval = warpSums[lane];
        unsigned int wactive = __ballot_sync(0xffffffffu, lane < nWarps);
        unsigned int wex = warpExclusiveScan(wval, lane, wactive);
        warpSums[lane] = wex;
    }
    __syncthreads();

    // Step 4: add warp prefix to each thread's exclusive scan
    unsigned int warpPrefix = warpSums[warpId];
    unsigned int scanVal = ex + warpPrefix;
    as[k] = scanVal;
    __syncthreads();
    // ----------------------------------------------------------------

    if (k == blockDim.x - 1) {
        outidx[blockIdx.x] = as[k] + codewordlen;
        kcmax = (as[k] + codewordlen) >> 5; // divide by 32
    }

    // Write the codes
    kc = as[k] >> 5;            // /32
    startbit = as[k] & 31;      // %32
    as[k] = 0U;
    __syncthreads();

    // Precompute masks to reduce control divergence and operations
    unsigned int remaining = codewordlen;

    // Part 1
    wrbits = remaining > (32u - startbit) ? (32u - startbit) : remaining;
    if (wrbits) {
        tmpcw32 = (unsigned int)(cw64 >> (remaining - wrbits));
        atomicOr(&as[kc], tmpcw32 << (32u - startbit - wrbits));
        remaining -= wrbits;
    }

    // Part 2
    if (remaining) {
        wrbits = remaining > 32u ? 32u : remaining;
        tmpcw32 = (unsigned int)(cw64 >> (remaining - wrbits)) &
                  ((wrbits == 32u) ? 0xffffffffu : ((1u << wrbits) - 1u));
        atomicOr(&as[kc + 1], tmpcw32 << (32u - wrbits));
        remaining -= wrbits;
    }

    // Part 3
    if (remaining) {
        tmpcw32 = (unsigned int)(cw64 & ((remaining == 32u) ? 0xffffffffu
                                                            : ((1u << remaining) - 1u)));
        atomicOr(&as[kc + 2], tmpcw32 << (32u - remaining));
    }

    __syncthreads();

    if (k <= kcmax)
        out[kn] = as[k];
}
//////////////////////////////////////////////////////////////////////////////
#endif

#endif
