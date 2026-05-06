#include "srad.h"
#include <stdio.h>

__global__ void srad_cuda_1(float * __restrict__ E_C,
                            float * __restrict__ W_C,
                            float * __restrict__ N_C,
                            float * __restrict__ S_C,
                            float * __restrict__ J_cuda,
                            float * __restrict__ C_cuda,
                            int cols,
                            int rows,
                            float q0sqr) {

    // block id
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // thread id
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // base index for this block
    const int block_base = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx;

    // indices
    const int index   = block_base + cols * ty + tx;
    const int index_n = index - cols;
    const int index_s = block_base + cols * BLOCK_SIZE + tx;
    const int index_w = index - 1;
    const int index_e = block_base + cols * ty + BLOCK_SIZE;

    float n, w, e, s, jc, g2, l, num, den, qsqr, c;

    // shared memory allocation
    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float temp_result[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ float north[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float south[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float east[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float west[BLOCK_SIZE][BLOCK_SIZE];

    // load north/south to shared memory with coalesced accesses
    north[ty][tx] = J_cuda[index_n];
    south[ty][tx] = J_cuda[index_s];

    if (by == 0) {
        north[ty][tx] = J_cuda[BLOCK_SIZE * bx + tx];
    } else if (by == gridDim.y - 1) {
        south[ty][tx] = J_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) +
                               BLOCK_SIZE * bx + cols * (BLOCK_SIZE - 1) + tx];
    }
    __syncthreads();

    // load west/east to shared memory with coalesced accesses
    west[ty][tx] = J_cuda[index_w];
    east[ty][tx] = J_cuda[index_e];

    if (bx == 0) {
        west[ty][tx] = J_cuda[cols * BLOCK_SIZE * by + cols * ty];
    } else if (bx == gridDim.x - 1) {
        east[ty][tx] =
            J_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * (gridDim.x - 1) +
                   cols * ty + BLOCK_SIZE - 1];
    }

    __syncthreads();

    // center pixel
    jc = J_cuda[index];
    temp[ty][tx] = jc;

    __syncthreads();

    // compute directional differences with reduced branching
    if (ty == 0) {
        n = north[ty][tx] - jc;
    } else {
        n = temp[ty - 1][tx] - jc;
    }

    if (ty == BLOCK_SIZE - 1) {
        s = south[ty][tx] - jc;
    } else {
        s = temp[ty + 1][tx] - jc;
    }

    if (tx == 0) {
        w = west[ty][tx] - jc;
    } else {
        w = temp[ty][tx - 1] - jc;
    }

    if (tx == BLOCK_SIZE - 1) {
        e = east[ty][tx] - jc;
    } else {
        e = temp[ty][tx + 1] - jc;
    }

    // avoid division by zero
    float jc_sq = jc * jc;
    jc_sq = (jc_sq > 1e-12f) ? jc_sq : 1e-12f;

    g2 = (n * n + s * s + w * w + e * e) / jc_sq;
    l  = (n + s + w + e) / jc;

    num  = 0.5f * g2 - (0.0625f * l * l);     // 0.0625 = 1/16
    den  = 1.0f + 0.25f * l;
    qsqr = num / (den * den);

    // diffusion coefficient (equ 33)
    float q0sqr_loc = q0sqr;
    float base = q0sqr_loc * (1.0f + q0sqr_loc);
    float inv_base = 1.0f / base;

    den = (qsqr - q0sqr_loc) * inv_base;
    c = 1.0f / (1.0f + den);

    // saturate diffusion coefficient without extra branches
    c = fminf(1.0f, fmaxf(0.0f, c));
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
                            int cols,
                            int rows,
                            float lambda,
                            float q0sqr) {
    // block id
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // thread id
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // base index for this block
    const int block_base = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx;

    // indices
    const int index   = block_base + cols * ty + tx;
    const int index_s = block_base + cols * BLOCK_SIZE + tx;
    const int index_e = block_base + cols * ty + BLOCK_SIZE;

    float cc, cn, cs, ce, cw, d_sum;

    // shared memory allocation
    __shared__ float south_c[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float east_c[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ float c_cuda_temp[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float c_cuda_result[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE];

    // load data to shared memory (center)
    float j_val = J_cuda[index];
    temp[ty][tx] = j_val;

    __syncthreads();

    // load south coefficients
    south_c[ty][tx] = C_cuda[index_s];

    if (by == gridDim.y - 1) {
        south_c[ty][tx] =
            C_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) + BLOCK_SIZE * bx +
                   cols * (BLOCK_SIZE - 1) + tx];
    }
    __syncthreads();

    // load east coefficients
    east_c[ty][tx] = C_cuda[index_e];

    if (bx == gridDim.x - 1) {
        east_c[ty][tx] =
            C_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * (gridDim.x - 1) +
                   cols * ty + BLOCK_SIZE - 1];
    }

    __syncthreads();

    // center diffusion coefficients
    cc = C_cuda[index];
    c_cuda_temp[ty][tx] = cc;

    __syncthreads();

    // neighbor diffusion coefficients with simplified branching
    cn = cc;
    cw = cc;

    if (ty == BLOCK_SIZE - 1) {
        cs = south_c[ty][tx];
    } else {
        cs = c_cuda_temp[ty + 1][tx];
    }

    if (tx == BLOCK_SIZE - 1) {
        ce = east_c[ty][tx];
    } else {
        ce = c_cuda_temp[ty][tx + 1];
    }

    // divergence (equ 58)
    d_sum =
        cn * N_C[index] + cs * S_C[index] + cw * W_C[index] + ce * E_C[index];

    // image update (equ 61)
    c_cuda_result[ty][tx] = j_val + 0.25f * lambda * d_sum;

    __syncthreads();

    J_cuda[index] = c_cuda_result[ty][tx];
}
