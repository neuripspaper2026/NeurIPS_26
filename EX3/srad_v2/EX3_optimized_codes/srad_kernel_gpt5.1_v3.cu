#include "srad.h"
#include <stdio.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

// Use launch_bounds to guide compiler for A100 (tune threads-per-block & min blocks per SM as needed)
__global__ __launch_bounds__(BLOCK_SIZE * BLOCK_SIZE, 2)
void srad_cuda_1(float *__restrict__ E_C,
                 float *__restrict__ W_C,
                 float *__restrict__ N_C,
                 float *__restrict__ S_C,
                 float *__restrict__ J_cuda,
                 float *__restrict__ C_cuda,
                 int cols, int rows,
                 float q0sqr) {

    // block id
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // thread id
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // precompute common base index for this block
    const int block_base = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    const int row_offset = cols * ty;
    const int index = block_base + row_offset + tx;

    // neighbor indices
    const int index_n = index - cols;
    const int index_s = index + cols * BLOCK_SIZE;
    const int index_w = index - 1;
    const int index_e = index + BLOCK_SIZE;

    float n, w, e, s, jc, g2, l, num, den, qsqr, c;

    // shared memory allocation (flattened for better addressing)
    __shared__ float temp[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float temp_result[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float north[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float south[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float east[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float west[BLOCK_SIZE * BLOCK_SIZE];

    const int s_idx = ty * BLOCK_SIZE + tx;

    // load data to shared memory - north/south
    float north_val = J_cuda[index_n];
    float south_val = J_cuda[index_s];

    if (by == 0) {
        north_val = J_cuda[BLOCK_SIZE * bx + tx];
    } else if (by == gridDim.y - 1) {
        north_val = J_cuda[index_n]; // keep in-bounds; already correct above
        south_val = J_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) +
                           BLOCK_SIZE * bx + cols * (BLOCK_SIZE - 1) + tx];
    }

    north[s_idx] = north_val;
    south[s_idx] = south_val;
    __syncthreads();

    // load data to shared memory - west/east
    float west_val = J_cuda[index_w];
    float east_val = J_cuda[index_e];

    if (bx == 0) {
        west_val = J_cuda[cols * BLOCK_SIZE * by + cols * ty];
    } else if (bx == gridDim.x - 1) {
        east_val = J_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * (gridDim.x - 1) +
                          cols * ty + BLOCK_SIZE - 1];
    }

    west[s_idx] = west_val;
    east[s_idx] = east_val;
    __syncthreads();

    // center
    float center = J_cuda[index];
    temp[s_idx] = center;
    __syncthreads();

    jc = center;

    // compute neighbors using reduced branching
    if (ty == 0) {
        n = north[s_idx] - jc;
    } else {
        n = temp[(ty - 1) * BLOCK_SIZE + tx] - jc;
    }

    if (ty == BLOCK_SIZE - 1) {
        s = south[s_idx] - jc;
    } else {
        s = temp[(ty + 1) * BLOCK_SIZE + tx] - jc;
    }

    if (tx == 0) {
        w = west[s_idx] - jc;
    } else {
        w = temp[ty * BLOCK_SIZE + tx - 1] - jc;
    }

    if (tx == BLOCK_SIZE - 1) {
        e = east[s_idx] - jc;
    } else {
        e = temp[ty * BLOCK_SIZE + tx + 1] - jc;
    }

    // diffusion calculations
    const float jc2 = jc * jc;
    g2 = (n * n + s * s + w * w + e * e) / jc2;

    l = (n + s + w + e) / jc;

    num = 0.5f * g2 - (1.0f / 16.0f) * (l * l);
    den = 1.0f + 0.25f * l;
    qsqr = num / (den * den);

    // diffusion coefficient (equ 33)
    den = (qsqr - q0sqr) / (q0sqr * (1.0f + q0sqr));
    c = 1.0f / (1.0f + den);

    // saturate diffusion coefficient
    c = fminf(fmaxf(c, 0.0f), 1.0f);
    temp_result[s_idx] = c;

    __syncthreads();

    const float c_res = temp_result[s_idx];
    C_cuda[index] = c_res;
    E_C[index] = e;
    W_C[index] = w;
    S_C[index] = s;
    N_C[index] = n;
}

__global__ __launch_bounds__(BLOCK_SIZE * BLOCK_SIZE, 2)
void srad_cuda_2(float *__restrict__ E_C,
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

    // precompute common base index for this block
    const int block_base = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    const int row_offset = cols * ty;
    const int index = block_base + row_offset + tx;

    const int index_s = index + cols * BLOCK_SIZE;
    const int index_e = index + BLOCK_SIZE;

    float cc, cn, cs, ce, cw, d_sum;

    // shared memory allocation (flattened)
    __shared__ float south_c[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float east_c[BLOCK_SIZE * BLOCK_SIZE];

    __shared__ float c_cuda_temp[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float c_cuda_result[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float temp[BLOCK_SIZE * BLOCK_SIZE];

    const int s_idx = ty * BLOCK_SIZE + tx;

    // load data to shared memory
    float center = J_cuda[index];
    temp[s_idx] = center;

    __syncthreads();

    float south_val = C_cuda[index_s];
    if (by == gridDim.y - 1) {
        south_val =
            C_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) + BLOCK_SIZE * bx +
                   cols * (BLOCK_SIZE - 1) + tx];
    }
    south_c[s_idx] = south_val;
    __syncthreads();

    float east_val = C_cuda[index_e];
    if (bx == gridDim.x - 1) {
        east_val =
            C_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * (gridDim.x - 1) +
                   cols * ty + BLOCK_SIZE - 1];
    }
    east_c[s_idx] = east_val;

    __syncthreads();

    float cc_local = C_cuda[index];
    c_cuda_temp[s_idx] = cc_local;

    __syncthreads();

    cc = cc_local;

    // neighbor coefficients with minimized branching
    cn = cc;

    if (ty == BLOCK_SIZE - 1) {
        cs = south_c[s_idx];
    } else {
        cs = c_cuda_temp[(ty + 1) * BLOCK_SIZE + tx];
    }

    cw = cc;

    if (tx == BLOCK_SIZE - 1) {
        ce = east_c[s_idx];
    } else {
        ce = c_cuda_temp[ty * BLOCK_SIZE + tx + 1];
    }

    // divergence (equ 58)
    d_sum = cn * N_C[index] + cs * S_C[index] + cw * W_C[index] + ce * E_C[index];

    // image update (equ 61)
    c_cuda_result[s_idx] = center + 0.25f * lambda * d_sum;

    __syncthreads();

    J_cuda[index] = c_cuda_result[s_idx];
}
