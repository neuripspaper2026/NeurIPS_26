#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>

// Removed deprecated texture declarations
// texture<float4, 1, cudaReadModeElementType> tex;
// texture<float, 1, cudaReadModeElementType> txt;

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

__global__ void mergeSortFirst(float4 *result, int listsize, cudaTextureObject_t tex) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int limit = listsize >> 2;

    if (gtid >= limit)
        return;

    float4 r = tex1Dfetch<float4>(tex, gtid);
    result[gtid] = sortElem(r);
}

__global__ void mergeSortPass(float4 *result, int nrElems, int threadsPerDiv, cudaTextureObject_t tex) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int division = tid / threadsPerDiv;
    if (division >= DIVISIONS)
        return;

    int int_tid = tid - division * threadsPerDiv;
    int Astart = constStartAddr[division] + int_tid * nrElems;
    int divisionEnd = constStartAddr[division + 1];

    if (Astart >= divisionEnd)
        return;

    int Bstart = Astart + (nrElems >> 1);
    float4 *resStart = &(result[Astart]);

    if (Bstart >= divisionEnd) {
        int elems = divisionEnd - Astart;
#pragma unroll 1
        for (int i = 0; i < elems; i++) {
            resStart[i] = tex1Dfetch<float4>(tex, Astart + i);
        }
        return;
    }

    int aidx = 0;
    int bidx = 0;
    int outidx = 0;

    float4 a = tex1Dfetch<float4>(tex, Astart + aidx);
    float4 b = tex1Dfetch<float4>(tex, Bstart + bidx);

    int halfElems = nrElems >> 1;

    while (true) {
        float4 nextA = tex1Dfetch<float4>(tex, Astart + aidx + 1);
        float4 nextB = tex1Dfetch<float4>(tex, Bstart + bidx + 1);

        float4 na = getLowest(a, b);
        float4 nb = getHighest(a, b);
        a = sortElem(na);
        b = sortElem(nb);

        resStart[outidx++] = a;

        bool elemsLeftInA =
            (aidx + 1 < halfElems);
        bool elemsLeftInB = (bidx + 1 < halfElems) &&
                            (Bstart + bidx + 1 < divisionEnd);

        if (elemsLeftInA) {
            if (elemsLeftInB) {
                if (nextA.x < nextB.x) {
                    aidx += 1;
                    a = nextA;
                } else {
                    bidx += 1;
                    a = nextB;
                }
            } else {
                aidx += 1;
                a = nextA;
            }
        } else {
            if (elemsLeftInB) {
                bidx += 1;
                a = nextB;
            } else {
                break;
            }
        }
    }

    resStart[outidx++] = b;
}

__global__ void mergepack(float *orig, float *result) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int division = blockIdx.y;

    int outPos = finalStartAddr[division] + idx;
    int outEnd = finalStartAddr[division + 1];

    if (outPos >= outEnd)
        return;

    int inPos = constStartAddr[division] * 4 + nullElems[division] + idx;
    result[outPos] = orig[inPos];
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
