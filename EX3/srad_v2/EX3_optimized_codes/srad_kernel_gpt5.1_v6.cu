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

    // indices
    const int base = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    const int index = base + cols * ty + tx;
    const int index_n = index - cols;
    const int index_s = base + cols * BLOCK_SIZE + tx;
    const int index_w = index - 1;
    const int index_e = base + cols * ty + BLOCK_SIZE;

    float n, w, e, s, jc, g2, l, num, den, qsqr, c;

    // shared memory allocation
    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float temp_result[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ float north[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float south[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float east[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float west[BLOCK_SIZE][BLOCK_SIZE];

    // load data to shared memory (use cached loads)
    north[ty][tx] = __ldg(&J_cuda[index_n]);
    south[ty][tx] = __ldg(&J_cuda[index_s]);

    if (by == 0) {
        north[ty][tx] = __ldg(&J_cuda[BLOCK_SIZE * bx + tx]);
    } else if (by == gridDim.y - 1) {
        south[ty][tx] = __ldg(&J_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) +
                                      BLOCK_SIZE * bx + cols * (BLOCK_SIZE - 1) + tx]);
    }
    __syncthreads();

    west[ty][tx] = __ldg(&J_cuda[index_w]);
    east[ty][tx] = __ldg(&J_cuda[index_e]);

    if (bx == 0) {
        west[ty][tx] = __ldg(&J_cuda[cols * BLOCK_SIZE * by + cols * ty]);
    } else if (bx == gridDim.x - 1) {
        east[ty][tx] = __ldg(&J_cuda[cols * BLOCK_SIZE * by +
                                     BLOCK_SIZE * (gridDim.x - 1) +
                                     cols * ty + BLOCK_SIZE - 1]);
    }

    __syncthreads();

    temp[ty][tx] = __ldg(&J_cuda[index]);

    __syncthreads();

    jc = temp[ty][tx];

    // compute neighbors with minimized branching
    const bool is_n = (ty == 0);
    const bool is_s = (ty == BLOCK_SIZE - 1);
    const bool is_w = (tx == 0);
    const bool is_e = (tx == BLOCK_SIZE - 1);

    if (is_n) {
        n = north[ty][tx] - jc;
    } else {
        n = temp[ty - 1][tx] - jc;
    }

    if (is_s) {
        s = south[ty][tx] - jc;
    } else {
        s = temp[ty + 1][tx] - jc;
    }

    if (is_w) {
        w = west[ty][tx] - jc;
    } else {
        w = temp[ty][tx - 1] - jc;
    }

    if (is_e) {
        e = east[ty][tx] - jc;
    } else {
        e = temp[ty][tx + 1] - jc;
    }

    const float jc2 = jc * jc;
    const float n2 = n * n;
    const float s2 = s * s;
    const float w2 = w * w;
    const float e2 = e * e;

    g2 = (n2 + s2 + w2 + e2) / jc2;
    l = (n + s + w + e) / jc;

    num = 0.5f * g2 - (0.0625f) * (l * l);       // 1/16 = 0.0625
    den = 1.0f + 0.25f * l;
    qsqr = num / (den * den);

    // diffusion coefficient (equ 33)
    den = (qsqr - q0sqr) / (q0sqr * (1.0f + q0sqr));
    c = 1.0f / (1.0f + den);

    // saturate diffusion coefficient using branchless clamps
    c = fminf(fmaxf(c, 0.0f), 1.0f);
    temp_result[ty][tx] = c;

    __syncthreads();

    const float c_val = temp_result[ty][tx];
    C_cuda[index] = c_val;
    E_C[index] = e;
    W_C[index] = w;
    S_C[index] = s;
    N_C[index] = n;
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

    // indices
    const int base = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    const int index = base + cols * ty + tx;
    const int index_s = base + cols * BLOCK_SIZE + tx;
    const int index_e = base + cols * ty + BLOCK_SIZE;
    float cc, cn, cs, ce, cw, d_sum;

    // shared memory allocation
    __shared__ float south_c[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float east_c[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ float c_cuda_temp[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float c_cuda_result[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE];

    // load data to shared memory
    temp[ty][tx] = __ldg(&J_cuda[index]);

    __syncthreads();

    south_c[ty][tx] = __ldg(&C_cuda[index_s]);

    if (by == gridDim.y - 1) {
        south_c[ty][tx] =
            __ldg(&C_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) +
                          BLOCK_SIZE * bx +
                          cols * (BLOCK_SIZE - 1) + tx]);
    }
    __syncthreads();

    east_c[ty][tx] = __ldg(&C_cuda[index_e]);

    if (bx == gridDim.x - 1) {
        east_c[ty][tx] =
            __ldg(&C_cuda[cols * BLOCK_SIZE * by +
                          BLOCK_SIZE * (gridDim.x - 1) +
                          cols * ty + BLOCK_SIZE - 1]);
    }

    __syncthreads();

    c_cuda_temp[ty][tx] = __ldg(&C_cuda[index]);

    __syncthreads();

    cc = c_cuda_temp[ty][tx];

    const bool is_s = (ty == BLOCK_SIZE - 1);
    const bool is_e = (tx == BLOCK_SIZE - 1);

    cn = cc;
    cw = cc;

    if (is_s) {
        cs = south_c[ty][tx];
    } else {
        cs = c_cuda_temp[ty + 1][tx];
    }

    if (is_e) {
        ce = east_c[ty][tx];
    } else {
        ce = c_cuda_temp[ty][tx + 1];
    }

    // divergence (equ 58)
    d_sum =
        cn * __ldg(&N_C[index]) +
        cs * __ldg(&S_C[index]) +
        cw * __ldg(&W_C[index]) +
        ce * __ldg(&E_C[index]);

    // image update (equ 61)
    c_cuda_result[ty][tx] = temp[ty][tx] + 0.25f * lambda * d_sum;

    __syncthreads();

    J_cuda[index] = c_cuda_result[ty][tx];
}
