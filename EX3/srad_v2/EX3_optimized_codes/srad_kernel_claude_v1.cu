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
    jc = J_cuda[index];
    temp[ty][tx] = jc;

    // Load north neighbor
    if (by == 0 && ty == 0) {
        n = J_cuda[BLOCK_SIZE * bx + tx];
    } else if (ty == 0) {
        n = J_cuda[index_n];
    } else {
        n = temp[ty - 1][tx];
    }
    
    // Load south neighbor
    if (by == gridDim.y - 1 && ty == BLOCK_SIZE - 1) {
        s = J_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) +
                   BLOCK_SIZE * bx + cols * (BLOCK_SIZE - 1) + tx];
    } else if (ty == BLOCK_SIZE - 1) {
        s = J_cuda[index_s];
    } else {
        __syncthreads();
        s = temp[ty + 1][tx];
    }

    // Load west neighbor
    if (bx == 0 && tx == 0) {
        w = J_cuda[cols * BLOCK_SIZE * by + cols * ty];
    } else if (tx == 0) {
        w = J_cuda[index_w];
    } else {
        w = temp[ty][tx - 1];
    }

    // Load east neighbor
    if (bx == gridDim.x - 1 && tx == BLOCK_SIZE - 1) {
        e = J_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * (gridDim.x - 1) +
                   cols * ty + BLOCK_SIZE - 1];
    } else if (tx == BLOCK_SIZE - 1) {
        e = J_cuda[index_e];
    } else {
        e = temp[ty][tx + 1];
    }

    // Compute differences
    n -= jc;
    s -= jc;
    w -= jc;
    e -= jc;

    // Compute g2 and l using fused multiply-add
    float n2 = n * n;
    float s2 = s * s;
    float w2 = w * w;
    float e2 = e * e;
    
    float jc_inv = 1.0f / jc;
    g2 = (n2 + s2 + w2 + e2) * jc_inv * jc_inv;
    l = (n + s + w + e) * jc_inv;

    // Compute diffusion coefficient
    float l2 = l * l;
    num = fmaf(0.5f, g2, -0.0625f * l2);
    den = fmaf(0.25f, l, 1.0f);
    qsqr = num / (den * den);

    den = (qsqr - q0sqr) / fmaf(q0sqr, q0sqr, q0sqr);
    c = 1.0f / fmaf(den, 1.0f, 1.0f);

    // saturate diffusion coefficient
    c = fmaxf(0.0f, fminf(1.0f, c));

    // Store results with coalesced writes
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
    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE];

    // load data to shared memory with coalesced access
    float j_val = J_cuda[index];
    temp[ty][tx] = j_val;
    
    cc = C_cuda[index];
    c_cuda_temp[ty][tx] = cc;

    __syncthreads();

    // Load south coefficient
    if (by == gridDim.y - 1 && ty == BLOCK_SIZE - 1) {
        cs = C_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) + BLOCK_SIZE * bx +
                    cols * (BLOCK_SIZE - 1) + tx];
    } else if (ty == BLOCK_SIZE - 1) {
        cs = C_cuda[index_s];
    } else {
        cs = c_cuda_temp[ty + 1][tx];
    }

    // Load east coefficient
    if (bx == gridDim.x - 1 && tx == BLOCK_SIZE - 1) {
        ce = C_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * (gridDim.x - 1) +
                    cols * ty + BLOCK_SIZE - 1];
    } else if (tx == BLOCK_SIZE - 1) {
        ce = C_cuda[index_e];
    } else {
        ce = c_cuda_temp[ty][tx + 1];
    }

    cn = cc;
    cw = cc;

    // Load directional differences with coalesced reads
    float n_val = N_C[index];
    float s_val = S_C[index];
    float w_val = W_C[index];
    float e_val = E_C[index];

    // divergence computation using fused multiply-add
    d_sum = fmaf(cn, n_val, fmaf(cs, s_val, fmaf(cw, w_val, ce * e_val)));

    // image update
    float result = fmaf(0.25f * lambda, d_sum, j_val);

    // Direct write without intermediate shared memory
    J_cuda[index] = result;
}
