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

    unsigned int kn = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = threadIdx.x;
    unsigned int kc, startbit, wrbits;

    unsigned long long cw64 = 0;
    unsigned int val32, codewordlen = 0;
    unsigned char tmpbyte, tmpcwlen;
    unsigned int tmpcw32;

    extern __shared__ unsigned int sm[];
    __shared__ unsigned int kcmax;

#ifdef CACHECWLUT
    unsigned int *codewords = (unsigned int *)sm;
    unsigned int *codewordlens = (unsigned int *)(sm + NUM_SYMBOLS);
    unsigned int *as = (unsigned int *)(sm + 2 * NUM_SYMBOLS);

    /* Load the codewords and the original data with vectorized access */
    if (k < NUM_SYMBOLS) {
        uint2 cw_data = *reinterpret_cast<const uint2*>(&gm_codewords[k & ~1]);
        uint2 cwlen_data = *reinterpret_cast<const uint2*>(&gm_codewordlens[k & ~1]);
        if (k & 1) {
            codewords[k] = cw_data.y;
            codewordlens[k] = cwlen_data.y;
        } else {
            codewords[k] = cw_data.x;
            codewordlens[k] = cwlen_data.x;
        }
    }
    val32 = data[kn];
    __syncthreads();
    
    /* Unroll the byte processing loop */
    #pragma unroll
    for (unsigned int i = 0; i < 4; i++) {
        tmpbyte = (unsigned char)(val32 >> ((3 - i) * 8));
        tmpcw32 = codewords[tmpbyte];
        tmpcwlen = codewordlens[tmpbyte];
        cw64 = (cw64 << tmpcwlen) | tmpcw32;
        codewordlen += tmpcwlen;
    }
#else
    unsigned int *as = (unsigned int *)sm;
    val32 = data[kn];
    
    /* Unroll the byte processing loop */
    #pragma unroll
    for (unsigned int i = 0; i < 4; i++) {
        tmpbyte = (unsigned char)(val32 >> ((3 - i) * 8));
        tmpcw32 = gm_codewords[tmpbyte];
        tmpcwlen = gm_codewordlens[tmpbyte];
        cw64 = (cw64 << tmpcwlen) | tmpcw32;
        codewordlen += tmpcwlen;
    }
#endif
    as[k] = codewordlen;
    __syncthreads();

    /* Prefix sum using warp-level primitives for A100 */
    unsigned int lane = threadIdx.x & 31;
    unsigned int warpId = threadIdx.x >> 5;
    unsigned int numWarps = blockDim.x >> 5;
    
    /* Warp-level scan */
    unsigned int warp_sum = as[k];
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        unsigned int temp = __shfl_up_sync(0xFFFFFFFF, warp_sum, offset);
        if (lane >= offset) warp_sum += temp;
    }
    
    __shared__ unsigned int warp_sums[32];
    
    /* Store warp sums */
    if (lane == 31) {
        warp_sums[warpId] = warp_sum;
    }
    __syncthreads();
    
    /* Scan warp sums (single warp operation) */
    if (threadIdx.x < numWarps) {
        unsigned int sum = warp_sums[threadIdx.x];
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            unsigned int temp = __shfl_up_sync(0xFFFFFFFF, sum, offset);
            if (threadIdx.x >= offset) sum += temp;
        }
        warp_sums[threadIdx.x] = sum;
    }
    __syncthreads();
    
    /* Convert to exclusive scan */
    unsigned int block_sum = (warpId > 0) ? warp_sums[warpId - 1] : 0;
    unsigned int exclusive_scan = block_sum + warp_sum - as[k];
    as[k] = exclusive_scan;
    __syncthreads();

    if (k == blockDim.x - 1) {
        outidx[blockIdx.x] = exclusive_scan + codewordlen;
        kcmax = (exclusive_scan + codewordlen) / 32;
    }
    __syncthreads();

    /* Write the codes with reduced atomic contention */
    kc = exclusive_scan / 32;
    startbit = exclusive_scan % 32;
    
    /* Initialize output buffer efficiently */
    if (k <= kcmax) {
        as[k] = 0U;
    }
    __syncthreads();

    /* Part 1 - optimized with reduced branching */
    wrbits = min(codewordlen, 32 - startbit);
    if (wrbits > 0) {
        tmpcw32 = (unsigned int)(cw64 >> (codewordlen - wrbits));
        atomicOr(&as[kc], tmpcw32 << (32 - startbit - wrbits));
        codewordlen -= wrbits;
    }

    /* Part 2 - optimized with predication */
    if (codewordlen > 0) {
        wrbits = min(codewordlen, 32U);
        tmpcw32 = (unsigned int)(cw64 >> (codewordlen - wrbits)) & ((1U << wrbits) - 1);
        atomicOr(&as[kc + 1], tmpcw32 << (32 - wrbits));
        codewordlen -= wrbits;
    }

    /* Part 3 - optimized with predication */
    if (codewordlen > 0) {
        tmpcw32 = (unsigned int)(cw64 & ((1U << codewordlen) - 1));
        atomicOr(&as[kc + 2], tmpcw32 << (32 - codewordlen));
    }

    __syncthreads();

    /* Coalesced write to global memory */
    if (k <= kcmax) {
        out[kn] = as[k];
    }
}
//////////////////////////////////////////////////////////////////////////////
#endif

#endif
