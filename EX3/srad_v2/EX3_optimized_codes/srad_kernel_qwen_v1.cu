<<<CODE>>>
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

    // shared memory allocation
    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float temp_result[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ float north[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float south[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float east[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float west[BLOCK_SIZE][BLOCK_SIZE];

    // Load data to shared memory with coalesced access pattern
    // Load north and south boundaries
    north[ty][tx] = (by == 0) ? J_cuda[BLOCK_SIZE * bx + tx] : J_cuda[index_n];
    south[ty][tx] = (by == gridDim.y - 1) ? 
        J_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) + BLOCK_SIZE * bx + cols * (BLOCK_SIZE - 1) + tx] : 
        J_cuda[index_s];
    
    __syncthreads();

    // Load west and east boundaries
    west[ty][tx] = (bx == 0) ? J_cuda[cols * BLOCK_SIZE * by + cols * ty] : J_cuda[index_w];
    east[ty][tx] = (bx == gridDim.x - 1) ? 
        J_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * (gridDim.x - 1) + cols * ty + BLOCK_SIZE - 1] : 
        J_cuda[index_e];

    __syncthreads();

    temp[ty][tx] = J_cuda[index];
    __syncthreads();

    jc = temp[ty][tx];

    // Boundary handling with optimized branching
    if (ty == 0) {
        n = north[ty][tx] - jc;
        s = temp[ty + 1][tx] - jc;
    } else if (ty == BLOCK_SIZE - 1) {
        n = temp[ty - 1][tx] - jc;
        s = south[ty][tx] - jc;
    } else {
        n = temp[ty - 1][tx] - jc;
        s = temp[ty + 1][tx] - jc;
    }

    if (tx == 0) {
        w = west[ty][tx] - jc;
        e = temp[ty][tx + 1] - jc;
    } else if (tx == BLOCK_SIZE - 1) {
        w = temp[ty][tx - 1] - jc;
        e = east[ty][tx] - jc;
    } else {
        w = temp[ty][tx - 1] - jc;
        e = temp[ty][tx + 1] - jc;
    }

    // Corner cases
    if (ty == 0 && tx == 0) { // nw
        n = north[ty][tx] - jc;
        s = temp[ty + 1][tx] - jc;
        w = west[ty][tx] - jc;
        e = temp[ty][tx + 1] - jc;
    } else if (ty == 0 && tx == BLOCK_SIZE - 1) { // ne
        n = north[ty][tx] - jc;
        s = temp[ty + 1][tx] - jc;
        w = temp[ty][tx - 1] - jc;
        e = east[ty][tx] - jc;
    } else if (ty == BLOCK_SIZE - 1 && tx == BLOCK_SIZE - 1) { // se
        n = temp[ty - 1][tx] - jc;
        s = south[ty][tx] - jc;
        w = temp[ty][tx - 1] - jc;
        e = east[ty][tx] - jc;
    } else if (ty == BLOCK_SIZE - 1 && tx == 0) { // sw
        n = temp[ty - 1][tx] - jc;
        s = south[ty][tx] - jc;
        w = west[ty][tx] - jc;
        e = temp[ty][tx + 1] - jc;
    }

    g2 = (n * n + s * s + w * w + e * e) / (jc * jc);
    l = (n + s + w + e) / jc;

    num = (0.5f * g2) - ((1.0f / 16.0f) * (l * l));
    den = 1.0f + (0.25f * l);
    qsqr = num / (den * den);

    // diffusion coefficient (equ 33)
    den = (qsqr - q0sqr) / (q0sqr * (1.0f + q0sqr));
    c = 1.0f / (1.0f + den);

    // saturate diffusion coefficient
    c = fmaxf(0.0f, fminf(1.0f, c));

    temp_result[ty][tx] = c;
    __syncthreads();

    C_cuda[index] = temp_result[ty][tx];
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

    // shared memory allocation
    __shared__ float south_c[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float east_c[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ float c_cuda_temp[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float c_cuda_result[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE];

    // Load data to shared memory with coalesced access
    temp[ty][tx] = J_cuda[index];
    __syncthreads();

    south_c[ty][tx] = (by == gridDim.y - 1) ? 
        C_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) + BLOCK_SIZE * bx + cols * (BLOCK_SIZE - 1) + tx] : 
        C_cuda[index_s];
    __syncthreads();

    east_c[ty][tx] = (bx == gridDim.x - 1) ? 
        C_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * (gridDim.x - 1) + cols * ty + BLOCK_SIZE - 1] : 
        C_cuda[index_e];
    __syncthreads();

    c_cuda_temp[ty][tx] = C_cuda[index];
    __syncthreads();

    cc = c_cuda_temp[ty][tx];

    // Optimized boundary handling
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

    cn = cc;
    cw = cc;

    // Handle corner case explicitly
    if (ty == BLOCK_SIZE - 1 && tx == BLOCK_SIZE - 1) { // se
        cs = south_c[ty][tx];
        ce = east_c[ty][tx];
    }

    // divergence (equ 58)
    d_sum = cn * N_C[index] + cs * S_C[index] + cw * W_C[index] + ce * E
