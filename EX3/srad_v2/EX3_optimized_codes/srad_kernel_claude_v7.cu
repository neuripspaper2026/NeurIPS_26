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
    
    // Prefetch boundary conditions
    bool is_top = (by == 0);
    bool is_bottom = (by == gridDim.y - 1);
    bool is_left = (bx == 0);
    bool is_right = (bx == gridDim.x - 1);
    
    __syncthreads();

    jc = temp[ty][tx];

    // Compute directional differences using on-demand loading
    // North neighbor
    if (ty == 0) {
        if (is_top) {
            n = J_cuda[BLOCK_SIZE * bx + tx] - jc;
        } else {
            n = J_cuda[index_n] - jc;
        }
    } else {
        n = temp[ty - 1][tx] - jc;
    }

    // South neighbor
    if (ty == BLOCK_SIZE - 1) {
        if (is_bottom) {
            n = temp[ty - 1][tx] - jc;
            s = J_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) +
                       BLOCK_SIZE * bx + cols * (BLOCK_SIZE - 1) + tx] - jc;
        } else {
            s = J_cuda[index_s] - jc;
        }
    } else {
        s = temp[ty + 1][tx] - jc;
    }

    // West neighbor
    if (tx == 0) {
        if (is_left) {
            w = J_cuda[cols * BLOCK_SIZE * by + cols * ty] - jc;
        } else {
            w = J_cuda[index_w] - jc;
        }
    } else {
        w = temp[ty][tx - 1] - jc;
    }

    // East neighbor
    if (tx == BLOCK_SIZE - 1) {
        if (is_right) {
            e = J_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * (gridDim.x - 1) +
                       cols * ty + BLOCK_SIZE - 1] - jc;
        } else {
            e = J_cuda[index_e] - jc;
        }
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

    // saturate diffusion coefficent
    c = fminf(fmaxf(c, 0.0f), 1.0f);

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
    temp[ty][tx] = J_cuda[index];
    c_cuda_temp[ty][tx] = C_cuda[index];
    
    bool is_bottom = (by == gridDim.y - 1);
    bool is_right = (bx == gridDim.x - 1);

    __syncthreads();

    cc = c_cuda_temp[ty][tx];

    // Compute coefficients with on-demand loading
    cn = cc;
    cw = cc;
    
    // South coefficient
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

    // East coefficient
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

    // divergence (equ 58) - use FMA for better performance
    d_sum = cn * N_C[index] + cs * S_C[index] + cw * W_C[index] + ce * E_C[index];

    // image update (equ 61) - use FMA
    float result = fmaf(0.25f * lambda, d_sum, temp[ty][tx]);

    // Direct write without intermediate shared memory
    J_cuda[index] = result;
}
