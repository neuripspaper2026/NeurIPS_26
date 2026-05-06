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

    /* Load the codewords and the original data with coalesced access */
    if (k < NUM_SYMBOLS) {
        codewords[k] = __ldg(&gm_codewords[k]);
        codewordlens[k] = __ldg(&gm_codewordlens[k]);
    }
    val32 = __ldg(&data[kn]);
    __syncthreads();
    
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
    val32 = __ldg(&data[kn]);
    
    #pragma unroll
    for (unsigned int i = 0; i < 4; i++) {
        tmpbyte = (unsigned char)(val32 >> ((3 - i) * 8));
        tmpcw32 = __ldg(&gm_codewords[tmpbyte]);
        tmpcwlen = __ldg(&gm_codewordlens[tmpbyte]);
        cw64 = (cw64 << tmpcwlen) | tmpcw32;
        codewordlen += tmpcwlen;
    }
#endif
    as[k] = codewordlen;
    __syncthreads();

    /* Prefix sum using warp-level primitives for better performance on A100 */
    unsigned int lane = k & 31;
    unsigned int warpid = k >> 5;
    unsigned int numwarps = (blockDim.x + 31) >> 5;
    
    /* Warp-level scan */
    unsigned int warp_sum = codewordlen;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        unsigned int n = __shfl_up_sync(0xffffffff, warp_sum, offset);
        if (lane >= offset) warp_sum += n;
    }
    
    __shared__ unsigned int warp_sums[32];
    if (lane == 31) {
        warp_sums[warpid] = warp_sum;
    }
    __syncthreads();
    
    /* Scan warp sums */
    if (warpid == 0 && lane < numwarps) {
        unsigned int sum = warp_sums[lane];
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            unsigned int n = __shfl_up_sync(0xffffffff, sum, offset);
            if (lane >= offset) sum += n;
        }
        warp_sums[lane] = sum;
    }
    __syncthreads();
    
    /* Compute exclusive prefix sum */
    unsigned int block_sum = (warpid > 0) ? warp_sums[warpid - 1] : 0;
    unsigned int exclusive_sum = block_sum + warp_sum - codewordlen;
    as[k] = exclusive_sum;
    __syncthreads();

    if (k == blockDim.x - 1) {
        unsigned int total_bits = exclusive_sum + codewordlen;
        outidx[blockIdx.x] = total_bits;
        kcmax = total_bits / 32;
    }
    __syncthreads();

    /* Write the codes */
    kc = as[k] / 32;
    startbit = as[k] % 32;
    
    /* Initialize output locations */
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

    /* Part 2 */
    if (codewordlen > 0) {
        wrbits = min(codewordlen, 32U);
        tmpcw32 = (unsigned int)(cw64 >> (codewordlen - wrbits)) & ((1U << wrbits) - 1);
        atomicOr(&as[kc + 1], tmpcw32 << (32 - wrbits));
        codewordlen -= wrbits;
    }

    /* Part 3 */
    if (codewordlen > 0) {
        tmpcw32 = (unsigned int)(cw64 & ((1U << codewordlen) - 1));
        atomicOr(&as[kc + 2], tmpcw32 << (32 - codewordlen));
    }

    __syncthreads();

    if (k <= kcmax) {
        out[kn] = as[k];
    }
}
//////////////////////////////////////////////////////////////////////////////
#endif

#endif
