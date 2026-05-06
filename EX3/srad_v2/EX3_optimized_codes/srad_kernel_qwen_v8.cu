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

    // shared memory allocation with padding to avoid bank conflicts
    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float temp_result[BLOCK_SIZE][BLOCK_SIZE + 1];

    __shared__ float north[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float south[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float east[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float west[BLOCK_SIZE][BLOCK_SIZE + 1];

    // load data to shared memory with coalesced access
    north[ty][tx] = J_cuda[index_n];
    south[ty][tx] = J_cuda[index_s];
    if (by == 0) {
        north[ty][tx] = J_cuda[BLOCK_SIZE * bx + tx];
    } else if (by == gridDim.y - 1) {
        south[ty][tx] = J_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) +
                               BLOCK_SIZE * bx + cols * (BLOCK_SIZE - 1) + tx];
    }
    __syncthreads();

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

    temp[ty][tx] = J_cuda[index];

    __syncthreads();

    jc = temp[ty][tx];

    // Simplified boundary condition handling to reduce divergence
    bool is_north = (ty == 0);
    bool is_south = (ty == BLOCK_SIZE - 1);
    bool is_west = (tx == 0);
    bool is_east = (tx == BLOCK_SIZE - 1);

    n = (is_north) ? north[ty][tx] - jc : temp[ty - 1][tx] - jc;
    s = (is_south) ? south[ty][tx] - jc : temp[ty + 1][tx] - jc;
    w = (is_west) ? west[ty][tx] - jc : temp[ty][tx - 1] - jc;
    e = (is_east) ? east[ty][tx] - jc : temp[ty][tx + 1] - jc;

    g2 = (n * n + s * s + w * w + e * e) / (jc * jc);

    l = (n + s + w + e) / jc;

    num = (0.5f * g2) - ((1.0f / 16.0f) * (l * l));
    den = 1.0f + (0.25f * l);
    qsqr = num / (den * den);

    // diffusion coefficent (equ 33)
    den = (qsqr - q0sqr) / (q0sqr * (1.0f + q0sqr));
    c = 1.0f / (1.0f + den);

    // saturate diffusion coefficent with predicated stores
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

    // shared memory allocation with padding to avoid bank conflicts
    __shared__ float south_c[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float east_c[BLOCK_SIZE][BLOCK_SIZE + 1];

    __shared__ float c_cuda_temp[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float c_cuda_result[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE + 1];

    // load data to shared memory
    temp[ty][tx] = J_cuda[index];

    __syncthreads();

    south_c[ty][tx] = C_cuda[index_s];

    if (by == gridDim.y - 1) {
        south_c[ty][tx] =
            C_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) + BLOCK_SIZE * bx +
                   cols * (BLOCK_SIZE - 1) + tx];
    }
    __syncthreads();

    east_c[ty][tx] = C_cuda[index_e];

    if (bx == gridDim.x - 1) {
        east_c[ty][tx] =
            C_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * (gridDim.x - 1) +
                   cols * ty + BLOCK_SIZE - 1];
    }

    __syncthreads();

    c_cuda_temp[ty][tx] = C_cuda[index];

    __syncthreads();

    cc = c_cuda_temp[ty][tx];

    // Simplified boundary handling
    bool is_south = (ty == BLOCK_SIZE - 1);
    bool is_east = (tx == BLOCK_SIZE - 1);

    cn = cc;
    cw = cc;
    cs = (is_south) ? south_c[ty][tx] : c_cuda_temp[ty + 1][tx];
    ce = (is_east) ? east_c[ty][tx] : c_cuda_temp[ty][tx + 1];

    // divergence (equ 58)
    d_sum =
        cn * N_C[index] + cs * S_C[index] + cw * W_C[index] + ce * E_C[index];

    // image update (equ 61)
    c_cuda_result[ty][tx] = temp[ty][tx] + 0.25f * lambda * d_sum;

    __syncthreads();

    J_cuda[index] = c_cuda_result[ty][tx];
}
