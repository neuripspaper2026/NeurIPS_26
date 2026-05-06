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
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // thread id
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // precompute common terms
    const int block_col_offset = cols * BLOCK_SIZE * by;
    const int block_offset     = block_col_offset + BLOCK_SIZE * bx;
    const int row_offset       = cols * ty;
    const int index            = block_offset + row_offset + tx;

    const int index_n = index - cols;
    const int index_s = index + cols * BLOCK_SIZE;
    const int index_w = index - 1;
    const int index_e = index + BLOCK_SIZE;

    float n, w, e, s, jc, g2, l, num, den, qsqr, c;

    // shared memory allocation (avoid bank conflicts via padding on x-dim)
    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float temp_result[BLOCK_SIZE][BLOCK_SIZE + 1];

    __shared__ float north[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float south[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float east[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float west[BLOCK_SIZE][BLOCK_SIZE + 1];

    // load data to shared memory: coalesced loads using computed indices
    if (by == 0) {
        north[ty][tx] = J_cuda[BLOCK_SIZE * bx + tx];
    } else {
        north[ty][tx] = J_cuda[index_n];
    }

    if (by == gridDim.y - 1) {
        south[ty][tx] = J_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) +
                               BLOCK_SIZE * bx + cols * (BLOCK_SIZE - 1) + tx];
    } else {
        south[ty][tx] = J_cuda[index_s];
    }
    __syncthreads();

    if (bx == 0) {
        west[ty][tx] = J_cuda[block_col_offset + row_offset];
    } else {
        west[ty][tx] = J_cuda[index_w];
    }

    if (bx == gridDim.x - 1) {
        east[ty][tx] =
            J_cuda[block_col_offset + BLOCK_SIZE * (gridDim.x - 1) +
                   row_offset + BLOCK_SIZE - 1];
    } else {
        east[ty][tx] = J_cuda[index_e];
    }

    __syncthreads();

    temp[ty][tx] = J_cuda[index];

    __syncthreads();

    jc = temp[ty][tx];

    // compute neighbors using reduced branching on borders

    const bool is_north = (ty == 0);
    const bool is_south = (ty == BLOCK_SIZE - 1);
    const bool is_west  = (tx == 0);
    const bool is_east  = (tx == BLOCK_SIZE - 1);

    float n_val, s_val, w_val, e_val;

    // north neighbor
    n_val = is_north ? north[ty][tx] : temp[ty - 1][tx];
    // south neighbor
    s_val = is_south ? south[ty][tx] : temp[ty + 1][tx];
    // west neighbor
    w_val = is_west ? west[ty][tx] : temp[ty][tx - 1];
    // east neighbor
    e_val = is_east ? east[ty][tx] : temp[ty][tx + 1];

    n = n_val - jc;
    s = s_val - jc;
    w = w_val - jc;
    e = e_val - jc;

    // use FMA-friendly expressions
    const float jc2 = jc * jc;
    g2 = (n * n + s * s + w * w + e * e) / jc2;

    l = (n + s + w + e) / jc;

    num = 0.5f * g2 - (0.0625f * l * l); // 1/16 = 0.0625
    den = 1.0f + 0.25f * l;
    qsqr = num / (den * den);

    // diffusion coefficient (equ 33)
    den = (qsqr - q0sqr) / (q0sqr * (1.0f + q0sqr));
    c = 1.0f / (1.0f + den);

    // saturate diffusion coefficient using min/max to reduce branching
    c = c < 0.0f ? 0.0f : c;
    c = c > 1.0f ? 1.0f : c;
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
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // thread id
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // precompute common terms
    const int block_col_offset = cols * BLOCK_SIZE * by;
    const int block_offset     = block_col_offset + BLOCK_SIZE * bx;
    const int row_offset       = cols * ty;
    const int index            = block_offset + row_offset + tx;

    const int index_s = index + cols * BLOCK_SIZE;
    const int index_e = index + BLOCK_SIZE;
    float cc, cn, cs, ce, cw, d_sum;

    // shared memory allocation (avoid bank conflicts via padding)
    __shared__ float south_c[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float east_c[BLOCK_SIZE][BLOCK_SIZE + 1];

    __shared__ float c_cuda_temp[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float c_cuda_result[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE + 1];

    // load data to shared memory
    temp[ty][tx] = J_cuda[index];

    __syncthreads();

    if (by == gridDim.y - 1) {
        south_c[ty][tx] =
            C_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) + BLOCK_SIZE * bx +
                   cols * (BLOCK_SIZE - 1) + tx];
    } else {
        south_c[ty][tx] = C_cuda[index_s];
    }
    __syncthreads();

    if (bx == gridDim.x - 1) {
        east_c[ty][tx] =
            C_cuda[block_col_offset + BLOCK_SIZE * (gridDim.x - 1) +
                   row_offset + BLOCK_SIZE - 1];
    } else {
        east_c[ty][tx] = C_cuda[index_e];
    }

    __syncthreads();

    c_cuda_temp[ty][tx] = C_cuda[index];

    __syncthreads();

    cc = c_cuda_temp[ty][tx];

    const bool is_south = (ty == BLOCK_SIZE - 1);
    const bool is_east  = (tx == BLOCK_SIZE - 1);

    // current coefficient is reused
    cn = cc;
    cw = cc;

    // south neighbor coefficient
    cs = is_south ? south_c[ty][tx] : c_cuda_temp[ty + 1][tx];
    // east neighbor coefficient
    ce = is_east ? east_c[ty][tx] : c_cuda_temp[ty][tx + 1];

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
