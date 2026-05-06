#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>

__device__ __forceinline__ float4 sortElem(float4 r) {
    float4 nr;

    nr.x = (r.x > r.y) ? r.y : r.x;
    nr.y = (r.y > r.x) ? r.y : r.x;
    nr.z = (r.z > r.w) ? r.w : r.z;
    nr.w = (r.w > r.z) ? r.w : r.z;

    r.x = (nr.x > nr.z) ? nr.z : nr.x;
    r.y = (nr.y > nr.w) ? nr.w : nr.y;
    r.z = (nr.z > nr.x) ? nr.z : nr.x;
    r.w = (nr.w > nr.y) ? nr.w : nr.y;

    nr.x = r.x;
    nr.y = (r.y > r.z) ? r.z : r.y;
    nr.z = (r.z > r.y) ? r.z : r.y;
    nr.w = r.w;
    return nr;
}

__device__ __forceinline__ float4 getLowest(float4 a, float4 b) {
    a.x = (a.x < b.w) ? a.x : b.w;
    a.y = (a.y < b.z) ? a.y : b.z;
    a.z = (a.z < b.y) ? a.z : b.y;
    a.w = (a.w < b.x) ? a.w : b.x;
    return a;
}

__device__ __forceinline__ float4 getHighest(float4 a, float4 b) {
    b.x = (a.w >= b.x) ? a.w : b.x;
    b.y = (a.z >= b.y) ? a.z : b.y;
    b.z = (a.y >= b.z) ? a.y : b.z;
    b.w = (a.x >= b.w) ? a.x : b.w;
    return b;
}

__constant__ int constStartAddr[DIVISIONS + 1];
__constant__ int finalStartAddr[DIVISIONS + 1];
__constant__ int nullElems[DIVISIONS];

__global__ void mergeSortFirst(float4 * __restrict__ result,
                               int listsize,
                               cudaTextureObject_t tex) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int n4  = listsize >> 2;

    if (gid < n4) {
        float4 r = tex1Dfetch<float4>(tex, gid);
        result[gid] = sortElem(r);
    }
}

__global__ void mergeSortPass(float4 * __restrict__ result,
                              int nrElems,
                              int threadsPerDiv,
                              cudaTextureObject_t tex) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int division = tid / threadsPerDiv;

    if (division >= DIVISIONS)
        return;

    const int int_tid = tid - division * threadsPerDiv;
    const int Astart  = constStartAddr[division] + int_tid * nrElems;
    const int Aend    = constStartAddr[division + 1];

    if (Astart >= Aend)
        return;

    const int Bstart = Astart + (nrElems >> 1);
    float4 * __restrict__ resStart = result + Astart;

    if (Bstart >= Aend) {
#pragma unroll 4
        for (int i = 0; i < (Aend - Astart); ++i) {
            resStart[i] = tex1Dfetch<float4>(tex, Astart + i);
        }
        return;
    }

    int aidx = 0;
    int bidx = 0;
    int outidx = 0;

    float4 a = tex1Dfetch<float4>(tex, Astart + aidx);
    float4 b = tex1Dfetch<float4>(tex, Bstart + bidx);

    const int halfElems = nrElems >> 1;

    while (true) {
        const bool elemsLeftInA =
            (aidx + 1 < halfElems);
        const bool elemsLeftInB =
            (bidx + 1 < halfElems) && (Bstart + bidx + 1 < Aend);

        float4 nextA;
        float4 nextB;

        if (elemsLeftInA) {
            nextA = tex1Dfetch<float4>(tex, Astart + aidx + 1);
        } else {
            nextA = a;
        }

        if (elemsLeftInB) {
            nextB = tex1Dfetch<float4>(tex, Bstart + bidx + 1);
        } else {
            nextB = b;
        }

        float4 na = getLowest(a, b);
        float4 nb = getHighest(a, b);
        a = sortElem(na);
        b = sortElem(nb);

        resStart[outidx++] = a;

        if (elemsLeftInA) {
            if (elemsLeftInB) {
                if (nextA.x < nextB.x) {
                    ++aidx;
                    a = nextA;
                } else {
                    ++bidx;
                    a = nextB;
                }
            } else {
                ++aidx;
                a = nextA;
            }
        } else {
            if (elemsLeftInB) {
                ++bidx;
                a = nextB;
            } else {
                break;
            }
        }
    }
    resStart[outidx] = b;
}

__global__ void mergepack(float * __restrict__ orig,
                          float * __restrict__ result) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int division = blockIdx.y;

    const int outPos = finalStartAddr[division] + idx;

    if (outPos >= finalStartAddr[division + 1])
        return;

    const int inPos =
        constStartAddr[division] * 4 + nullElems[division] + idx;

    result[outPos] = orig[inPos];
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
