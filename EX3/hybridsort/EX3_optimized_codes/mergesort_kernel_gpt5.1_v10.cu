#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>

// Removed deprecated texture declarations
// texture<float4, 1, cudaReadModeElementType> tex;
// texture<float, 1, cudaReadModeElementType> txt;

__device__ __forceinline__ float4 sortElem(float4 r) {
    float4 nr;

    // First compare-exchange on (x,y) and (z,w)
    float ax = r.x;
    float ay = r.y;
    float az = r.z;
    float aw = r.w;

    float mn1 = fminf(ax, ay);
    float mx1 = fmaxf(ax, ay);
    float mn2 = fminf(az, aw);
    float mx2 = fmaxf(az, aw);

    nr.x = mn1;
    nr.y = mx1;
    nr.z = mn2;
    nr.w = mx2;

    // Cross compare-exchange between pairs
    ax = nr.x;
    ay = nr.y;
    az = nr.z;
    aw = nr.w;

    float mn3 = fminf(ax, az);
    float mn4 = fminf(ay, aw);
    float mx3 = fmaxf(az, ax);
    float mx4 = fmaxf(aw, ay);

    r.x = mn3;
    r.y = mn4;
    r.z = mx3;
    r.w = mx4;

    // Final compare-exchange on middle elements
    float by = r.y;
    float bz = r.z;
    float mn5 = fminf(by, bz);
    float mx5 = fmaxf(by, bz);

    nr.x = r.x;
    nr.y = mn5;
    nr.z = mx5;
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
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int limit = listsize >> 2; // listsize / 4

    // Grid-stride loop to better utilize SMs on large inputs
    for (int idx = gtid; idx < limit; idx += blockDim.x * gridDim.x) {
        float4 r = tex1Dfetch<float4>(tex, idx);
        result[idx] = sortElem(r);
    }
}

__global__ void mergeSortPass(float4 * __restrict__ result,
                              int nrElems,
                              int threadsPerDiv,
                              cudaTextureObject_t tex) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    int division = tid / threadsPerDiv;
    if (division >= DIVISIONS)
        return;

    int int_tid = tid - division * threadsPerDiv;
    int Astart = constStartAddr[division] + int_tid * nrElems;
    int Bstart = Astart + (nrElems >> 1);

    if (Astart >= constStartAddr[division + 1])
        return;

    float4 * __restrict__ resStart = &(result[Astart]);

    if (Bstart >= constStartAddr[division + 1]) {
        int end = constStartAddr[division + 1];
        int count = end - Astart;
        for (int i = 0; i < count; ++i) {
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
    const int divisionEnd = constStartAddr[division + 1];

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
        bool elemsLeftInB =
            (bidx + 1 < halfElems) &&
            (Bstart + bidx + 1 < divisionEnd);

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
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int division = blockIdx.y;

    int srcBase = constStartAddr[division] * 4 + nullElems[division];
    int dstBase = finalStartAddr[division];

    int gid = dstBase + idx;
    if (gid >= finalStartAddr[division + 1])
        return;

    result[gid] = orig[srcBase + idx];
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
