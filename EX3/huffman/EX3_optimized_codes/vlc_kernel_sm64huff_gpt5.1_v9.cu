#ifndef _VLC_SM64HUFF_KERNEL_H_
#define _VLC_SM64HUFF_KERNEL_H_

#include "pabio_kernels_v2.cu"
#include "parameters.h"

#ifdef SMATOMICS

// Warp-level inclusive scan for 32-bit values
__device__ __forceinline__ unsigned int warpInclusiveScan(unsigned int val) {
    const unsigned int FULL_MASK = 0xffffffffu;
    // Tree-based warp scan
    val += __shfl_up_sync(FULL_MASK, val, 1);
    val += __shfl_up_sync(FULL_MASK, val, 2);
    val += __shfl_up_sync(FULL_MASK, val, 4);
    val += __shfl_up_sync(FULL_MASK, val, 8);
    val += __shfl_up_sync(FULL_MASK, val, 16);
    return val;
}

// Block-wide exclusive scan using warp-level primitives and shared memory
__device__ __forceinline__ unsigned int blockExclusiveScan(unsigned int val,
                                                           unsigned int *smemWarpSums) {
    const int tid  = threadIdx.x;
    const int lane = tid & 31;         // lane within warp
    const int warp = tid >> 5;         // warp index within block

    unsigned int inclusive = warpInclusiveScan(val);
    unsigned int warpSum   = __shfl_sync(0xffffffffu, inclusive, 31);

    if (lane == 31) {
        smemWarpSums[warp] = warpSum;
    }
    __syncthreads();

    if (warp == 0) {
        unsigned int warpVal = (lane < (blockDim.x + 31) / 32) ? smemWarpSums[lane] : 0u;
        unsigned int warpScan = warpInclusiveScan(warpVal);
        if (lane < (blockDim.x + 31) / 32) {
            smemWarpSums[lane] = warpScan - warpVal;  // convert to exclusive for warps
        }
    }
    __syncthreads();

    unsigned int addend = smemWarpSums[warp];
    unsigned int exclusive = inclusive - val + addend;
    return exclusive;
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
    unsigned int val32, codewordlen = 0;
    unsigned char tmpbyte, tmpcwlen;
    unsigned int tmpcw32;

    extern __shared__ unsigned int sm[];
    __shared__ unsigned int kcmax;

#ifdef CACHECWLUT
    unsigned int *codewords    = (unsigned int *)sm;
    unsigned int *codewordlens = (unsigned int *)(sm + NUM_SYMBOLS);
    // reserve space for per-thread bit offsets plus warp-scan temps
    unsigned int *as           = (unsigned int *)(sm + 2 * NUM_SYMBOLS);
    // warp sums buffer placed after blockDim.x entries
    unsigned int *warpSums     = as + blockDim.x;

    // Load the codewords and the original data
    if (k < NUM_SYMBOLS) {
        codewords[k]    = gm_codewords[k];
        codewordlens[k] = gm_codewordlens[k];
    }
    __syncthreads();

    val32 = data[kn];

    // Unroll fixed 4-byte processing
#pragma unroll
    for (unsigned int i = 0; i < 4; i++) {
        tmpbyte  = static_cast<unsigned char>(val32 >> ((3 - i) * 8));
        tmpcw32  = codewords[tmpbyte];
        tmpcwlen = static_cast<unsigned char>(codewordlens[tmpbyte]);
        cw64     = (cw64 << tmpcwlen) | tmpcw32;
        codewordlen += tmpcwlen;
    }
#else
    unsigned int *as       = (unsigned int *)sm;
    unsigned int *warpSums = as + blockDim.x;

    val32 = data[kn];

#pragma unroll
    for (unsigned int i = 0; i < 4; i++) {
        tmpbyte  = static_cast<unsigned char>(val32 >> ((3 - i) * 8));
        tmpcw32  = gm_codewords[tmpbyte];
        tmpcwlen = static_cast<unsigned char>(gm_codewordlens[tmpbyte]);
        cw64     = (cw64 << tmpcwlen) | tmpcw32;
        codewordlen += tmpcwlen;
    }
#endif

    // Store local bit-length
    as[k] = codewordlen;
    __syncthreads();

    // Efficient block-wide exclusive scan of bit-lengths
    unsigned int bitOffset = blockExclusiveScan(as[k], warpSums);
    as[k] = bitOffset;

    __syncthreads();

    // Total bits for this block: last thread's offset + its own length
    if (k == blockDim.x - 1) {
        const unsigned int totalBits = as[k] + codewordlen;
        outidx[blockIdx.x] = totalBits;
        kcmax = totalBits >> 5; // /32
    }

    // Compute starting word and bit position
    kc       = bitOffset >> 5;       // /32
    startbit = bitOffset & 31u;      // %32
    as[k]    = 0u;
    __syncthreads();

    // Write the codes

    // Part 1
    wrbits = codewordlen > (32u - startbit) ? (32u - startbit) : codewordlen;
    tmpcw32 = static_cast<unsigned int>(cw64 >> (codewordlen - wrbits));
    atomicOr(&as[kc], tmpcw32 << (32u - startbit - wrbits));
    codewordlen -= wrbits;

    // Part 2
    if (codewordlen) {
        wrbits = codewordlen > 32u ? 32u : codewordlen;
        tmpcw32 = static_cast<unsigned int>(cw64 >> (codewordlen - wrbits)) &
                  ((1u << wrbits) - 1u);
        atomicOr(&as[kc + 1], tmpcw32 << (32u - wrbits));
        codewordlen -= wrbits;
    }

    // Part 3
    if (codewordlen) {
        tmpcw32 = static_cast<unsigned int>(cw64 & ((1ULL << codewordlen) - 1ULL));
        atomicOr(&as[kc + 2], tmpcw32 << (32u - codewordlen));
    }

    __syncthreads();

    // Coalesced write-back
    if (k <= kcmax)
        out[kn] = as[k];
}
//////////////////////////////////////////////////////////////////////////////
#endif

#endif
