#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>

__device__ __forceinline__ float4 sortElem(float4 r) {
    float4 nr;

    nr.x = fminf(r.x, r.y);
    nr.y = fmaxf(r.x, r.y);
    nr.z = fminf(r.z, r.w);
    nr.w = fmaxf(r.z, r.w);

    r.x = fminf(nr.x, nr.z);
    r.y = fminf(nr.y, nr.w);
    r.z = fmaxf(nr.x, nr.z);
    r.w = fmaxf(nr.y, nr.w);

    nr.x = r.x;
    nr.y = fminf(r.y, r.z);
    nr.z = fmaxf(r.y, r.z);
    nr.w = r.w;
    return nr;
}

__device__ __forceinline__ float4 getLowest(float4 a, float4 b) {
    a.x = fminf(a.x, b.w);
    a.y = fminf(a.y, b.z);
    a.z = fminf(a.z, b.y);
    a.w = fminf(a.w, b.x);
    return a;
}

__device__ __forceinline__ float4 getHighest(float4 a, float4 b) {
    b.x = fmaxf(a.w, b.x);
    b.y = fmaxf(a.z, b.y);
    b.z = fmaxf(a.y, b.z);
    b.w = fmaxf(a.x, b.w);
    return b;
}


__constant__ int constStartAddr[DIVISIONS + 1];
__constant__ int finalStartAddr[DIVISIONS + 1];
__constant__ int nullElems[DIVISIONS];

__global__ void __launch_bounds__(256, 4) mergeSortFirst(float4 *result, int listsize, cudaTextureObject_t tex) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elems = listsize / 4;
    
    if (idx < total_elems) {
        float4 r = tex1Dfetch<float4>(tex, idx);
        result[idx] = sortElem(r);
    }
}

__global__ void __launch_bounds__(256, 3) mergeSortPass(float4 *result, int nrElems, int threadsPerDiv, cudaTextureObject_t tex) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int division = tid / threadsPerDiv;
    
    if (division >= DIVISIONS)
        return;
    
    int int_tid = tid - division * threadsPerDiv;
    int Astart = constStartAddr[division] + int_tid * nrElems;
    int Bstart = Astart + nrElems / 2;
    int divEnd = constStartAddr[division + 1];
    
    if (Astart >= divEnd)
        return;
    
    float4 *resStart = &(result[Astart]);
    
    if (Bstart >= divEnd) {
        int copyCount = divEnd - Astart;
        #pragma unroll 2
        for (int i = 0; i < copyCount; i++) {
            resStart[i] = tex1Dfetch<float4>(tex, Astart + i);
        }
        return;
    }

    int aidx = 0;
    int bidx = 0;
    int outidx = 0;
    int halfElems = nrElems / 2;
    
    float4 a = tex1Dfetch<float4>(tex, Astart);
    float4 b = tex1Dfetch<float4>(tex, Bstart);

    while (true) {
        float4 nextA = tex1Dfetch<float4>(tex, Astart + aidx + 1);
        float4 nextB = tex1Dfetch<float4>(tex, Bstart + bidx + 1);

        float4 na = getLowest(a, b);
        float4 nb = getHighest(a, b);
        a = sortElem(na);
        b = sortElem(nb);
        
        resStart[outidx++] = a;

        bool elemsLeftInA = (aidx + 1 < halfElems);
        bool elemsLeftInB = (bidx + 1 < halfElems) && (Bstart + bidx + 1 < divEnd);

        if (elemsLeftInA) {
            if (elemsLeftInB) {
                bool chooseA = (nextA.x < nextB.x);
                aidx += chooseA;
                bidx += !chooseA;
                a = chooseA ? nextA : nextB;
            } else {
                aidx++;
                a = nextA;
            }
        } else {
            if (elemsLeftInB) {
                bidx++;
                a = nextB;
            } else {
                break;
            }
        }
    }
    resStart[outidx] = b;
}

__global__ void __launch_bounds__(256, 4) mergepack(float *orig, float *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int division = blockIdx.y;

    int startAddr = finalStartAddr[division];
    int endAddr = finalStartAddr[division + 1];
    
    if ((startAddr + idx) >= endAddr)
        return;
    
    int srcIdx = constStartAddr[division] * 4 + nullElems[division] + idx;
    result[startAddr + idx] = __ldg(&orig[srcIdx]);
}


#endif // #ifndef _MATRIXMUL_KERNEL_H_
