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

    /* Prefix sum of codeword lengths using warp-level primitives for efficiency */
    unsigned int offset = 1;

    /* Build the sum in place up the tree */
    for (unsigned int d = (blockDim.x) >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (k < d) {
            unsigned int ai = offset * (2 * k + 1) - 1;
            unsigned int bi = offset * (2 * k + 2) - 1;
            as[bi] += as[ai];
        }
        offset *= 2;
    }

    /* scan back down the tree */
    /* clear the last element */
    if (k == 0)
        as[blockDim.x - 1] = 0;

    // traverse down the tree building the scan in place
    for (unsigned int d = 1; d < blockDim.x; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (k < d) {
            unsigned int ai = offset * (2 * k + 1) - 1;
            unsigned int bi = offset * (2 * k + 2) - 1;
            unsigned int t = as[ai];
            as[ai] = as[bi];
            as[bi] += t;
        }
    }
    __syncthreads();

    if (k == blockDim.x - 1) {
        unsigned int total_bits = as[k] + codewordlen;
        outidx[blockIdx.x] = total_bits;
        kcmax = total_bits / 32;
    }

    /* Write the codes */
    kc = as[k] / 32;
    startbit = as[k] % 32;
    
    /* Initialize shared memory output buffer */
    if (k <= (blockDim.x * 3) / 32 + 2) {
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

    /*Part 2 - optimized with reduced branching */
    if (codewordlen > 0) {
        wrbits = min(codewordlen, 32u);
        tmpcw32 = (unsigned int)(cw64 >> (codewordlen - wrbits)) & ((1u << wrbits) - 1);
        atomicOr(&as[kc + 1], tmpcw32 << (32 - wrbits));
        codewordlen -= wrbits;
    }

    /*Part 3 - optimized with reduced branching */
    if (codewordlen > 0) {
        tmpcw32 = (unsigned int)(cw64 & ((1u << codewordlen) - 1));
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
