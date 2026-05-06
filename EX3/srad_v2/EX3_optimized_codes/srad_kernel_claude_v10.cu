#include "srad.h"
#include <stdio.h>

__global__ void srad_cuda_1(float *E_C, float *W_C, float *N_C, float *S_C,
                            float *J_cuda, float *C_cuda, int cols, int rows,
                            float q0sqr) {

    // block id
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // thread id
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // indices
    int index = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
    int index_n = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + tx - cols;
    int index_s =
        cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
    int index_w = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty - 1;
    int index_e =
        cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;

    float n, w, e, s, jc, g2, l, num, den, qsqr, c;

    // shared memory allocation - reduced from 6 to 2 arrays
    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float neighbors[BLOCK_SIZE][BLOCK_SIZE];

    // load center data to shared memory
    temp[ty][tx] = J_cuda[index];
    
    // Pre-compute boundary conditions
    bool is_top = (by == 0);
    bool is_bottom = (by == gridDim.y - 1);
    bool is_left = (bx == 0);
    bool is_right = (bx == gridDim.x - 1);
    
    __syncthreads();

    jc = temp[ty][tx];

    // Load and compute north neighbor
    if (ty == 0) {
        if (is_top) {
            n = J_cuda[BLOCK_SIZE * bx + tx] - jc;
        } else {
            n = J_cuda[index_n] - jc;
        }
    } else {
        n = temp[ty - 1][tx] - jc;
    }

    // Load and compute south neighbor
    if (ty == BLOCK_SIZE - 1) {
        if (is_bottom) {
            neighbors[ty][tx] = J_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) +
                                       BLOCK_SIZE * bx + cols * (BLOCK_SIZE - 1) + tx];
        } else {
            neighbors[ty][tx] = J_cuda[index_s];
        }
        s = neighbors[ty][tx] - jc;
    } else {
        s = temp[ty + 1][tx] - jc;
    }

    // Load and compute west neighbor
    if (tx == 0) {
        if (is_left) {
            w = J_cuda[cols * BLOCK_SIZE * by + cols * ty] - jc;
        } else {
            w = J_cuda[index_w] - jc;
        }
    } else {
        w = temp[ty][tx - 1] - jc;
    }

    // Load and compute east neighbor
    if (tx == BLOCK_SIZE - 1) {
        if (is_right) {
            neighbors[ty][tx] = J_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * (gridDim.x - 1) +
                                       cols * ty + BLOCK_SIZE - 1];
        } else {
            neighbors[ty][tx] = J_cuda[index_e];
        }
        e = neighbors[ty][tx] - jc;
    } else {
        e = temp[ty][tx + 1] - jc;
    }

    // Compute gradient and diffusion coefficient
    g2 = (n * n + s * s + w * w + e * e) / (jc * jc);
    l = (n + s + w + e) / jc;

    num = (0.5f * g2) - ((1.0f / 16.0f) * (l * l));
    den = 1.0f + (0.25f * l);
    qsqr = num / (den * den);

    // diffusion coefficent (equ 33)
    den = (qsqr - q0sqr) / (q0sqr * (1.0f + q0sqr));
    c = 1.0f / (1.0f + den);

    // saturate diffusion coefficent with fminf/fmaxf
    c = fminf(fmaxf(c, 0.0f), 1.0f);

    // Direct write to global memory (coalesced)
    C_cuda[index] = c;
    E_C[index] = e;
    W_C[index] = w;
    S_C[index] = s;
    N_C[index] = n;
}

__global__ void srad_cuda_2(float *E_C, float *W_C, float *N_C, float *S_C,
                            float *J_cuda, float *C_cuda, int cols, int rows,
                            float lambda, float q0sqr) {
    // block id
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // thread id
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // indices
    int index = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
    int index_s =
        cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
    int index_e =
        cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;
    float cc, cn, cs, ce, cw, d_sum;

    // shared memory allocation - reduced from 5 to 2 arrays
    __shared__ float c_cuda_temp[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float neighbors[BLOCK_SIZE][BLOCK_SIZE];

    // Pre-compute boundary conditions
    bool is_bottom = (by == gridDim.y - 1);
    bool is_right = (bx == gridDim.x - 1);

    // load data to shared memory
    float temp_j = J_cuda[index];
    c_cuda_temp[ty][tx] = C_cuda[index];

    __syncthreads();

    cc = c_cuda_temp[ty][tx];

    // Compute south coefficient
    if (ty == BLOCK_SIZE - 1) {
        if (is_bottom) {
            cs = C_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) + BLOCK_SIZE * bx +
                        cols * (BLOCK_SIZE - 1) + tx];
        } else {
            cs = C_cuda[index_s];
        }
    } else {
        cs = c_cuda_temp[ty + 1][tx];
    }

    // Compute east coefficient
    if (tx == BLOCK_SIZE - 1) {
        if (is_right) {
            ce = C_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * (gridDim.x - 1) +
                        cols * ty + BLOCK_SIZE - 1];
        } else {
            ce = C_cuda[index_e];
        }
    } else {
        ce = c_cuda_temp[ty][tx + 1];
    }

    cn = cc;
    cw = cc;

    // divergence (equ 58) - use fused multiply-add
    d_sum = __fmaf_rn(cn, N_C[index], __fmaf_rn(cs, S_C[index], 
            __fmaf_rn(cw, W_C[index], ce * E_C[index])));

    // image update (equ 61) - use fused multiply-add
    float result = __fmaf_rn(0.25f * lambda, d_sum, temp_j);

    // Direct write to global memory (coalesced)
    J_cuda[index] = result;
}
