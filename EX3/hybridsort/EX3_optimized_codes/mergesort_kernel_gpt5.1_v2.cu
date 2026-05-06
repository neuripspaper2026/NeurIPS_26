#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include <cuda_runtime.h>

// Removed deprecated texture declarations
// texture<float4, 1, cudaReadModeElementType> tex;
// texture<float, 1, cudaReadModeElementType> txt;

__device__ __forceinline__ float4 sortElem(float4 r) {
    float4 nr;

    // Use min/max intrinsics for better instruction selection on SM80+
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
    int nQuads = listsize >> 2;

    if (gtid >= nQuads)
        return;

    // Texture read is already cached; keep addressing arithmetic simple
    float4 r = tex1Dfetch<float4>(tex, gtid);
    result[gtid] = sortElem(r);
}

__global__ void mergeSortPass(float4 *result, int nrElems, int threadsPerDiv, cudaTextureObject_t tex) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int division = tid / threadsPerDiv;
    if (division >= DIVISIONS)
        return;

    int int_tid = tid - division * threadsPerDiv;
    int Astart = constStartAddr[division] + int_tid * nrElems;
    int Bstart = Astart + (nrElems >> 1);
    float4 *resStart = result + Astart;

    int divEnd = constStartAddr[division + 1];

    if (Astart >= divEnd)
        return;

    int validA = divEnd - Astart;
    if (Bstart >= divEnd) {
        // Tail copy, fully coalesced for threads in the same division
        for (int i = 0; i < validA; i++) {
            resStart[i] = tex1Dfetch<float4>(tex, Astart + i);
        }
        return;
    }

    int half = nrElems >> 1;

    int aidx = 0;
    int bidx = 0;
    int outidx = 0;

    float4 a = tex1Dfetch<float4>(tex, Astart + aidx);
    float4 b = tex1Dfetch<float4>(tex, Bstart + bidx);

    while (true) {
        // Prefetch using texture cache; indices are sequential within each half
        float4 nextA = tex1Dfetch<float4>(tex, Astart + aidx + 1);
        float4 nextB = tex1Dfetch<float4>(tex, Bstart + bidx + 1);

        float4 na = getLowest(a, b);
        float4 nb = getHighest(a, b);
        a = sortElem(na);
        b = sortElem(nb);

        resStart[outidx++] = a;

        bool elemsLeftInA = (aidx + 1) < half;
        bool elemsLeftInB = (bidx + 1 < half) && (Bstart + bidx + 1 < divEnd);

        if (elemsLeftInA) {
            if (elemsLeftInB) {
                // Select next run with branch-on-data; A100 has efficient predication
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
    resStart[outidx] = b;
}

__global__ void mergepack(float *orig, float *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int division = blockIdx.y;

    int start = finalStartAddr[division];
    int end = finalStartAddr[division + 1];

    int outIdx = start + idx;
    if (outIdx >= end)
        return;

    int inBase = (constStartAddr[division] << 2) + nullElems[division];
    result[outIdx] = orig[inBase + idx];
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
