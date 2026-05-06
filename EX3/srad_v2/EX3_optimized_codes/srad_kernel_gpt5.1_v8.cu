#include "srad.h"
#include <stdio.h>

__global__ void srad_cuda_1(float * __restrict__ E_C,
                            float * __restrict__ W_C,
                            float * __restrict__ N_C,
                            float * __restrict__ S_C,
                            float * __restrict__ J_cuda,
                            float * __restrict__ C_cuda,
                            int cols, int rows,
                            float q0sqr) {

    // block id
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // thread id
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const int bdim = BLOCK_SIZE;
    const int tileBase = cols * bdim * by + bdim * bx;
    const int rowBase  = tileBase + cols * ty;
    int index   = rowBase + tx;

    int index_n = index - cols;
    int index_s = index + cols * bdim;
    int index_w = index - 1;
    int index_e = index + bdim;

    float n, w, e, s, jc, g2, l, num, den, qsqr, c;

    // shared memory allocation
    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float temp_result[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ float north[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float south[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float east[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float west[BLOCK_SIZE][BLOCK_SIZE];

    const int lastBy = gridDim.y - 1;
    const int lastBx = gridDim.x - 1;

    // load data to shared memory - north/south
    float j_n = J_cuda[index_n];
    float j_s = J_cuda[index_s];

    if (by == 0) {
        j_n = J_cuda[bdim * bx + tx];
    } else if (by == lastBy) {
        j_s = J_cuda[cols * bdim * lastBy +
                     bdim * bx +
                     cols * (bdim - 1) + tx];
    }

    north[ty][tx] = j_n;
    south[ty][tx] = j_s;

    __syncthreads();

    // load data to shared memory - west/east
    float j_w = J_cuda[index_w];
    float j_e = J_cuda[index_e];

    if (bx == 0) {
        j_w = J_cuda[cols * bdim * by + cols * ty];
    } else if (bx == lastBx) {
        j_e = J_cuda[cols * bdim * by +
                     bdim * lastBx +
                     cols * ty + bdim - 1];
    }

    west[ty][tx] = j_w;
    east[ty][tx] = j_e;

    __syncthreads();

    // center value
    float j_center = J_cuda[index];
    temp[ty][tx] = j_center;

    __syncthreads();

    jc = temp[ty][tx];

    // compute directional differences with minimized branching
    if (ty == 0 && tx == 0) { // nw
        n = north[ty][tx] - jc;
        s = temp[ty + 1][tx] - jc;
        w = west[ty][tx] - jc;
        e = temp[ty][tx + 1] - jc;
    } else if (ty == 0 && tx == bdim - 1) { // ne
        n = north[ty][tx] - jc;
        s = temp[ty + 1][tx] - jc;
        w = temp[ty][tx - 1] - jc;
        e = east[ty][tx] - jc;
    } else if (ty == bdim - 1 && tx == bdim - 1) { // se
        n = temp[ty - 1][tx] - jc;
        s = south[ty][tx] - jc;
        w = temp[ty][tx - 1] - jc;
        e = east[ty][tx] - jc;
    } else if (ty == bdim - 1 && tx == 0) { // sw
        n = temp[ty - 1][tx] - jc;
        s = south[ty][tx] - jc;
        w = west[ty][tx] - jc;
        e = temp[ty][tx + 1] - jc;
    } else if (ty == 0) { // n
        n = north[ty][tx] - jc;
        s = temp[ty + 1][tx] - jc;
        w = temp[ty][tx - 1] - jc;
        e = temp[ty][tx + 1] - jc;
    } else if (tx == bdim - 1) { // e
        n = temp[ty - 1][tx] - jc;
        s = temp[ty + 1][tx] - jc;
        w = temp[ty][tx - 1] - jc;
        e = east[ty][tx] - jc;
    } else if (ty == bdim - 1) { // s
        n = temp[ty - 1][tx] - jc;
        s = south[ty][tx] - jc;
        w = temp[ty][tx - 1] - jc;
        e = temp[ty][tx + 1] - jc;
    } else if (tx == 0) { // w
        n = temp[ty - 1][tx] - jc;
        s = temp[ty + 1][tx] - jc;
        w = west[ty][tx] - jc;
        e = temp[ty][tx + 1] - jc;
    } else { // inner
        n = temp[ty - 1][tx] - jc;
        s = temp[ty + 1][tx] - jc;
        w = temp[ty][tx - 1] - jc;
        e = temp[ty][tx + 1] - jc;
    }

    // use FMA-friendly form to help compiler
    float jc2 = jc * jc;
    g2 = (n * n + s * s + w * w + e * e) / jc2;

    l = (n + s + w + e) / jc;

    num  = 0.5f * g2 - (1.0f / 16.0f) * (l * l);
    den  = 1.0f + 0.25f * l;
    qsqr = num / (den * den);

    // diffusion coefficent (equ 33)
    den = (qsqr - q0sqr) / (q0sqr * (1.0f + q0sqr));
    c   = 1.0f / (1.0f + den);

    // saturate diffusion coefficent (branch-lite via min/max)
    c = fminf(fmaxf(c, 0.0f), 1.0f);
    temp_result[ty][tx] = c;

    __syncthreads();

    C_cuda[index] = temp_result[ty][tx];
    E_C[index]    = e;
    W_C[index]    = w;
    S_C[index]    = s;
    N_C[index]    = n;
}

__global__ void srad_cuda_2(float * __restrict__ E_C,
                            float * __restrict__ W_C,
                            float * __restrict__ N_C,
                            float * __restrict__ S_C,
                            float * __restrict__ J_cuda,
                            float * __restrict__ C_cuda,
                            int cols, int rows,
                            float lambda, float q0sqr) {
    // block id
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // thread id
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const int bdim = BLOCK_SIZE;
    const int tileBase = cols * bdim * by + bdim * bx;
    const int rowBase  = tileBase + cols * ty;
    int index   = rowBase + tx;
    int index_s = index + cols * bdim;
    int index_e = index + bdim;

    float cc, cn, cs, ce, cw, d_sum;

    // shared memory allocation
    __shared__ float south_c[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float east_c[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ float c_cuda_temp[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float c_cuda_result[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE];

    const int lastBy = gridDim.y - 1;
    const int lastBx = gridDim.x - 1;

    // load data to shared memory
    float j_center = J_cuda[index];
    temp[ty][tx] = j_center;

    __syncthreads();

    float c_s = C_cuda[index_s];
    if (by == lastBy) {
        c_s = C_cuda[cols * bdim * lastBy +
                     bdim * bx +
                     cols * (bdim - 1) + tx];
    }
    south_c[ty][tx] = c_s;

    __syncthreads();

    float c_e = C_cuda[index_e];
    if (bx == lastBx) {
        c_e = C_cuda[cols * bdim * by +
                     bdim * lastBx +
                     cols * ty + bdim - 1];
    }
    east_c[ty][tx] = c_e;

    __syncthreads();

    float c_center = C_cuda[index];
    c_cuda_temp[ty][tx] = c_center;

    __syncthreads();

    cc = c_cuda_temp[ty][tx];

    if (ty == bdim - 1 && tx == bdim - 1) { // se
        cn = cc;
        cs = south_c[ty][tx];
        cw = cc;
        ce = east_c[ty][tx];
    } else if (tx == bdim - 1) { // e
        cn = cc;
        cs = c_cuda_temp[ty + 1][tx];
        cw = cc;
        ce = east_c[ty][tx];
    } else if (ty == bdim - 1) { // s
        cn = cc;
        cs = south_c[ty][tx];
        cw = cc;
        ce = c_cuda_temp[ty][tx + 1];
    } else { // inner
        cn = cc;
        cs = c_cuda_temp[ty + 1][tx];
        cw = cc;
        ce = c_cuda_temp[ty][tx + 1];
    }

    // divergence (equ 58)
    d_sum = cn * N_C[index] +
            cs * S_C[index] +
            cw * W_C[index] +
            ce * E_C[index];

    // image update (equ 61)
    c_cuda_result[ty][tx] = temp[ty][tx] + 0.25f * lambda * d_sum;

    __syncthreads();

    J_cuda[index] = c_cuda_result[ty][tx];
}
