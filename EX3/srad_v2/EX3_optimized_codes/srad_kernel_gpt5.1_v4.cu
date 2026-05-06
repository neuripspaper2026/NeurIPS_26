#include "srad.h"
#include <stdio.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

// Use __restrict__ to help the compiler with aliasing assumptions
__global__ void srad_cuda_1(float *__restrict__ E_C,
                            float *__restrict__ W_C,
                            float *__restrict__ N_C,
                            float *__restrict__ S_C,
                            float *__restrict__ J_cuda,
                            float *__restrict__ C_cuda,
                            int cols, int rows, float q0sqr) {

    // block id
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // thread id
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // global x,y
    const int x = bx * BLOCK_SIZE + tx;
    const int y = by * BLOCK_SIZE + ty;
    const int index = y * cols + x;

    // guard for partial blocks at borders
    if (x >= cols || y >= rows) return;

    const int north_y = (y == 0) ? 0 : y - 1;
    const int south_y = (y == rows - 1) ? rows - 1 : y + 1;
    const int west_x  = (x == 0) ? 0 : x - 1;
    const int east_x  = (x == cols - 1) ? cols - 1 : x + 1;

    const int index_n = north_y * cols + x;
    const int index_s = south_y * cols + x;
    const int index_w = y * cols + west_x;
    const int index_e = y * cols + east_x;

    float n, w, e, s, jc, g2, l, num, den, qsqr, c;

    // shared memory allocation (pad second dimension to reduce bank conflicts)
    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float temp_result[BLOCK_SIZE][BLOCK_SIZE + 1];

    __shared__ float north[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float south[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float east[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float west[BLOCK_SIZE][BLOCK_SIZE + 1];

    // load data to shared memory with clamped borders for entire image
    north[ty][tx] = J_cuda[index_n];
    south[ty][tx] = J_cuda[index_s];
    west[ty][tx]  = J_cuda[index_w];
    east[ty][tx]  = J_cuda[index_e];
    temp[ty][tx]  = J_cuda[index];

    __syncthreads();

    jc = temp[ty][tx];

    // compute neighbors using shared memory when inside tile,
    // otherwise use preloaded clamped halo values
    if (ty == 0) {
        n = north[ty][tx] - jc;
    } else {
        n = temp[ty - 1][tx] - jc;
    }

    if (ty == BLOCK_SIZE - 1 || y == rows - 1) {
        s = south[ty][tx] - jc;
    } else {
        s = temp[ty + 1][tx] - jc;
    }

    if (tx == 0) {
        w = west[ty][tx] - jc;
    } else {
        w = temp[ty][tx - 1] - jc;
    }

    if (tx == BLOCK_SIZE - 1 || x == cols - 1) {
        e = east[ty][tx] - jc;
    } else {
        e = temp[ty][tx + 1] - jc;
    }

    const float jc2 = jc * jc;
    g2 = (n * n + s * s + w * w + e * e) / jc2;

    l = (n + s + w + e) / jc;

    num  = 0.5f * g2 - (0.0625f * l * l);    // 1/16 = 0.0625
    den  = 1.0f + (0.25f * l);
    qsqr = num / (den * den);

    // diffusion coefficient (equ 33)
    den = (qsqr - q0sqr) / (q0sqr * (1.0f + q0sqr));
    c   = 1.0f / (1.0f + den);

    // clamp diffusion coefficient
    c = fminf(fmaxf(c, 0.0f), 1.0f);
    temp_result[ty][tx] = c;

    __syncthreads();

    C_cuda[index] = temp_result[ty][tx];
    E_C[index]    = e;
    W_C[index]    = w;
    S_C[index]    = s;
    N_C[index]    = n;
}

__global__ void srad_cuda_2(float *__restrict__ E_C,
                            float *__restrict__ W_C,
                            float *__restrict__ N_C,
                            float *__restrict__ S_C,
                            float *__restrict__ J_cuda,
                            float *__restrict__ C_cuda,
                            int cols, int rows,
                            float lambda, float q0sqr) {
    // block id
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // thread id
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // global x,y
    const int x = bx * BLOCK_SIZE + tx;
    const int y = by * BLOCK_SIZE + ty;
    const int index = y * cols + x;

    if (x >= cols || y >= rows) return;

    const int south_y = (y == rows - 1) ? rows - 1 : y + 1;
    const int east_x  = (x == cols - 1) ? cols - 1 : x + 1;

    const int index_s = south_y * cols + x;
    const int index_e = y * cols + east_x;

    float cc, cn, cs, ce, cw, d_sum;

    // shared memory allocation (pad to reduce bank conflicts)
    __shared__ float south_c[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float east_c[BLOCK_SIZE][BLOCK_SIZE + 1];

    __shared__ float c_cuda_temp[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float c_cuda_result[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE + 1];

    // load data to shared memory
    temp[ty][tx]      = J_cuda[index];
    south_c[ty][tx]   = C_cuda[index_s];
    east_c[ty][tx]    = C_cuda[index_e];
    c_cuda_temp[ty][tx] = C_cuda[index];

    __syncthreads();

    cc = c_cuda_temp[ty][tx];

    // select neighbors using shared memory for intra-tile, else halo (clamped)
    cn = cc;
    if (ty == BLOCK_SIZE - 1 || y == rows - 1) {
        cs = south_c[ty][tx];
    } else {
        cs = c_cuda_temp[ty + 1][tx];
    }

    cw = cc;
    if (tx == BLOCK_SIZE - 1 || x == cols - 1) {
        ce = east_c[ty][tx];
    } else {
        ce = c_cuda_temp[ty][tx + 1];
    }

    // divergence (equ 58)
    d_sum = cn * N_C[index] + cs * S_C[index] +
            cw * W_C[index] + ce * E_C[index];

    // image update (equ 61)
    c_cuda_result[ty][tx] = temp[ty][tx] + 0.25f * lambda * d_sum;

    __syncthreads();

    J_cuda[index] = c_cuda_result[ty][tx];
}
