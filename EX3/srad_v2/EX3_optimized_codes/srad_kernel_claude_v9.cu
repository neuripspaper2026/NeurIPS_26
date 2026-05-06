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

    // shared memory allocation - reduced from 6 to 5 arrays
    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float north[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float south[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float east[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float west[BLOCK_SIZE][BLOCK_SIZE];

    // load data to shared memory with coalesced access
    temp[ty][tx] = J_cuda[index];
    north[ty][tx] = J_cuda[index_n];
    south[ty][tx] = J_cuda[index_s];
    
    if (by == 0) {
        north[ty][tx] = J_cuda[BLOCK_SIZE * bx + tx];
    } else if (by == gridDim.y - 1) {
        south[ty][tx] = J_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) +
                               BLOCK_SIZE * bx + cols * (BLOCK_SIZE - 1) + tx];
    }
    
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

    jc = temp[ty][tx];

    // Optimized boundary condition handling with reduced branching
    bool is_top = (ty == 0);
    bool is_bottom = (ty == BLOCK_SIZE - 1);
    bool is_left = (tx == 0);
    bool is_right = (tx == BLOCK_SIZE - 1);

    n = (is_top ? north[ty][tx] : temp[ty - 1][tx]) - jc;
    s = (is_bottom ? south[ty][tx] : temp[ty + 1][tx]) - jc;
    w = (is_left ? west[ty][tx] : temp[ty][tx - 1]) - jc;
    e = (is_right ? east[ty][tx] : temp[ty][tx + 1]) - jc;

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

    // Direct write without intermediate shared memory
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

    // shared memory allocation - reduced from 5 to 3 arrays
    __shared__ float south_c[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float east_c[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float c_cuda_temp[BLOCK_SIZE][BLOCK_SIZE];

    // load data to shared memory with coalesced access
    float temp_j = J_cuda[index];
    c_cuda_temp[ty][tx] = C_cuda[index];
    south_c[ty][tx] = C_cuda[index_s];
    east_c[ty][tx] = C_cuda[index_e];

    if (by == gridDim.y - 1) {
        south_c[ty][tx] =
            C_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) + BLOCK_SIZE * bx +
                   cols * (BLOCK_SIZE - 1) + tx];
    }

    if (bx == gridDim.x - 1) {
        east_c[ty][tx] =
            C_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * (gridDim.x - 1) +
                   cols * ty + BLOCK_SIZE - 1];
    }

    __syncthreads();

    cc = c_cuda_temp[ty][tx];

    // Optimized boundary condition handling
    bool is_bottom = (ty == BLOCK_SIZE - 1);
    bool is_right = (tx == BLOCK_SIZE - 1);

    cn = cc;
    cs = is_bottom ? south_c[ty][tx] : c_cuda_temp[ty + 1][tx];
    cw = cc;
    ce = is_right ? east_c[ty][tx] : c_cuda_temp[ty][tx + 1];

    // divergence (equ 58)
    d_sum =
        cn * N_C[index] + cs * S_C[index] + cw * W_C[index] + ce * E_C[index];

    // image update (equ 61) - direct write without intermediate shared memory
    J_cuda[index] = temp_j + 0.25f * lambda * d_sum;
}
