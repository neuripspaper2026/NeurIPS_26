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

    // load north neighbor
    float north_val;
    if (by == 0) {
        north_val = J_cuda[BLOCK_SIZE * bx + tx];
    } else {
        north_val = J_cuda[index_n];
    }
    
    // load south neighbor
    float south_val;
    if (by == gridDim.y - 1) {
        south_val = J_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) +
                           BLOCK_SIZE * bx + cols * (BLOCK_SIZE - 1) + tx];
    } else {
        south_val = J_cuda[index_s];
    }

    // load west neighbor
    float west_val;
    if (bx == 0) {
        west_val = J_cuda[cols * BLOCK_SIZE * by + cols * ty];
    } else {
        west_val = J_cuda[index_w];
    }

    // load east neighbor
    float east_val;
    if (bx == gridDim.x - 1) {
        east_val = J_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * (gridDim.x - 1) +
                          cols * ty + BLOCK_SIZE - 1];
    } else {
        east_val = J_cuda[index_e];
    }

    __syncthreads();

    // Compute directional differences using shared memory and registers
    if (ty == 0) {
        n = north_val - jc;
    } else {
        n = temp[ty - 1][tx] - jc;
    }

    if (ty == BLOCK_SIZE - 1) {
        s = south_val - jc;
    } else {
        s = temp[ty + 1][tx] - jc;
    }

    if (tx == 0) {
        w = west_val - jc;
    } else {
        w = temp[ty][tx - 1] - jc;
    }

    if (tx == BLOCK_SIZE - 1) {
        e = east_val - jc;
    } else {
        e = temp[ty][tx + 1] - jc;
    }

    // Compute gradient and diffusion coefficient
    float jc_sq = jc * jc;
    g2 = (n * n + s * s + w * w + e * e) / jc_sq;
    l = (n + s + w + e) / jc;

    num = (0.5f * g2) - ((1.0f / 16.0f) * (l * l));
    den = 1.0f + (0.25f * l);
    qsqr = num / (den * den);

    // diffusion coefficent (equ 33)
    den = (qsqr - q0sqr) / (q0sqr * (1.0f + q0sqr));
    c = 1.0f / (1.0f + den);

    // saturate diffusion coefficent
    c = fminf(fmaxf(c, 0.0f), 1.0f);

    // Store results using coalesced writes
    neighbors[ty][tx] = c;
    __syncthreads();

    C_cuda[index] = neighbors[ty][tx];
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

    // load data to shared memory
    float j_val = J_cuda[index];
    temp[ty][tx] = j_val;

    // load south C value
    float south_c_val;
    if (by == gridDim.y - 1) {
        south_c_val = C_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) + BLOCK_SIZE * bx +
                             cols * (BLOCK_SIZE - 1) + tx];
    } else {
        south_c_val = C_cuda[index_s];
    }

    // load east C value
    float east_c_val;
    if (bx == gridDim.x - 1) {
        east_c_val = C_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * (gridDim.x - 1) +
                            cols * ty + BLOCK_SIZE - 1];
    } else {
        east_c_val = C_cuda[index_e];
    }

    cc = C_cuda[index];
    c_cuda_temp[ty][tx] = cc;

    __syncthreads();

    // Compute neighbor coefficients
    cn = cc;
    cw = cc;

    if (ty == BLOCK_SIZE - 1) {
        cs = south_c_val;
    } else {
        cs = c_cuda_temp[ty + 1][tx];
    }

    if (tx == BLOCK_SIZE - 1) {
        ce = east_c_val;
    } else {
        ce = c_cuda_temp[ty][tx + 1];
    }

    // Load directional differences
    float n_val = N_C[index];
    float s_val = S_C[index];
    float w_val = W_C[index];
    float e_val = E_C[index];

    // divergence (equ 58)
    d_sum = cn * n_val + cs * s_val + cw * w_val + ce * e_val;

    // image update (equ 61)
    float result = j_val + 0.25f * lambda * d_sum;

    __syncthreads();

    J_cuda[index] = result;
}
