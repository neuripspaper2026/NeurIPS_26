#include "srad.h"
#include <stdio.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

// Use __launch_bounds__ to help compiler optimize for occupancy on A100
__global__ __launch_bounds__(BLOCK_SIZE * BLOCK_SIZE, 2)
void srad_cuda_1(float * __restrict__ E_C,
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

    // global indices
    const int base = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    const int index = base + cols * ty + tx;
    const int index_n = index - cols;
    const int index_s = base + cols * BLOCK_SIZE + tx;
    const int index_w = index - 1;
    const int index_e = index + BLOCK_SIZE;

    float n, w, e, s, jc, g2, l, num, den, qsqr, c;

    // pad second dimension to avoid shared memory bank conflicts
    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float temp_result[BLOCK_SIZE][BLOCK_SIZE + 1];

    __shared__ float north[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float south[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float east[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float west[BLOCK_SIZE][BLOCK_SIZE + 1];

    // preload invariants
    const int last_by = gridDim.y - 1;
    const int last_bx = gridDim.x - 1;

    // load data to shared memory (north/south)
    float n_val = J_cuda[index_n];
    float s_val = J_cuda[index_s];

    if (by == 0) {
        n_val = J_cuda[BLOCK_SIZE * bx + tx];
    } else if (by == last_by) {
        const int base_last_row = cols * BLOCK_SIZE * last_by;
        n_val = J_cuda[index_n]; // keep for clarity; already loaded
        s_val = J_cuda[base_last_row + BLOCK_SIZE * bx +
                       cols * (BLOCK_SIZE - 1) + tx];
    }

    north[ty][tx] = n_val;
    south[ty][tx] = s_val;

    __syncthreads();

    // load data to shared memory (west/east)
    float w_val = J_cuda[index_w];
    float e_val = J_cuda[index_e];

    if (bx == 0) {
        w_val = J_cuda[cols * BLOCK_SIZE * by + cols * ty];
    } else if (bx == last_bx) {
        const int base_last_col = cols * BLOCK_SIZE * by + BLOCK_SIZE * last_bx;
        e_val = J_cuda[base_last_col + cols * ty + BLOCK_SIZE - 1];
    }

    west[ty][tx] = w_val;
    east[ty][tx] = e_val;

    __syncthreads();

    // center pixel
    jc = J_cuda[index];
    temp[ty][tx] = jc;

    __syncthreads();

    // compute directional differences with reduced branching
    const bool top    = (ty == 0);
    const bool bottom = (ty == BLOCK_SIZE - 1);
    const bool left   = (tx == 0);
    const bool right  = (tx == BLOCK_SIZE - 1);

    // north
    n = (top ? north[ty][tx] : temp[ty - 1][tx]) - jc;
    // south
    s = (bottom ? south[ty][tx] : temp[ty + 1][tx]) - jc;
    // west
    w = (left ? west[ty][tx] : temp[ty][tx - 1]) - jc;
    // east
    e = (right ? east[ty][tx] : temp[ty][tx + 1]) - jc;

    // gradient and Laplacian with fused multiply-adds
    const float jc2 = jc * jc;
    g2 = fmaf(n, n, fmaf(s, s, fmaf(w, w, e * e))) / jc2;
    l  = (n + s + w + e) / jc;

    num  = 0.5f * g2 - (0.0625f * l * l);       // 1/16 = 0.0625
    den  = 1.0f + 0.25f * l;
    qsqr = num / (den * den);

    // diffusion coefficient (equ 33)
    float t = (qsqr - q0sqr) / (q0sqr * (1.0f + q0sqr));
    c = 1.0f / (1.0f + t);

    // saturate diffusion coefficient using branchless clamps
    c = fminf(fmaxf(c, 0.0f), 1.0f);
    temp_result[ty][tx] = c;

    __syncthreads();

    // write back results
    C_cuda[index] = temp_result[ty][tx];
    E_C[index]    = e;
    W_C[index]    = w;
    S_C[index]    = s;
    N_C[index]    = n;
}

__global__ __launch_bounds__(BLOCK_SIZE * BLOCK_SIZE, 2)
void srad_cuda_2(float * __restrict__ E_C,
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

    // global indices
    const int base = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    const int index = base + cols * ty + tx;
    const int index_s = base + cols * BLOCK_SIZE + tx;
    const int index_e = index + BLOCK_SIZE;

    float cc, cn, cs, ce, cw, d_sum;

    // pad second dimension to avoid shared memory bank conflicts
    __shared__ float south_c[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float east_c[BLOCK_SIZE][BLOCK_SIZE + 1];

    __shared__ float c_cuda_temp[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float c_cuda_result[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE + 1];

    const int last_by = gridDim.y - 1;
    const int last_bx = gridDim.x - 1;

    // load image data to shared memory
    float j_val = J_cuda[index];
    temp[ty][tx] = j_val;

    __syncthreads();

    // south diffusion coefficients
    float cs_val = C_cuda[index_s];
    if (by == last_by) {
        const int base_last_row = cols * BLOCK_SIZE * last_by;
        cs_val = C_cuda[base_last_row + BLOCK_SIZE * bx +
                        cols * (BLOCK_SIZE - 1) + tx];
    }
    south_c[ty][tx] = cs_val;

    __syncthreads();

    // east diffusion coefficients
    float ce_val = C_cuda[index_e];
    if (bx == last_bx) {
        const int base_last_col = cols * BLOCK_SIZE * by + BLOCK_SIZE * last_bx;
        ce_val = C_cuda[base_last_col + cols * ty + BLOCK_SIZE - 1];
    }
    east_c[ty][tx] = ce_val;

    __syncthreads();

    // load central diffusion coefficient
    cc = C_cuda[index];
    c_cuda_temp[ty][tx] = cc;

    __syncthreads();

    const bool bottom = (ty == BLOCK_SIZE - 1);
    const bool right  = (tx == BLOCK_SIZE - 1);

    // neighbor diffusion coefficients
    cn = cc;
    cs = bottom ? south_c[ty][tx] : c_cuda_temp[ty + 1][tx];
    cw = cc;
    ce = right ? east_c[ty][tx] : c_cuda_temp[ty][tx + 1];

    // divergence (equ 58) using FMA where possible
    d_sum = fmaf(cn, N_C[index],
             fmaf(cs, S_C[index],
             fmaf(cw, W_C[index], ce * E_C[index])));

    // image update (equ 61)
    c_cuda_result[ty][tx] = j_val + 0.25f * lambda * d_sum;

    __syncthreads();

    J_cuda[index] = c_cuda_result[ty][tx];
}
