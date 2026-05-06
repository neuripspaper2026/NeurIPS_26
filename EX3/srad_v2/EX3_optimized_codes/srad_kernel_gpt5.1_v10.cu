#include "srad.h"
#include <stdio.h>

__global__ void srad_cuda_1(float * __restrict__ E_C, float * __restrict__ W_C,
                            float * __restrict__ N_C, float * __restrict__ S_C,
                            float * __restrict__ J_cuda,
                            float * __restrict__ C_cuda,
                            int cols, int rows, float q0sqr) {

    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int b_col = BLOCK_SIZE * bx;
    const int b_row = BLOCK_SIZE * by;

    const int g_col = b_col + tx;
    const int g_row = b_row + ty;

    const int index = g_row * cols + g_col;

    // Precompute useful values
    const int cols_block = cols * BLOCK_SIZE;
    const int gridDimx_minus1 = gridDim.x - 1;
    const int gridDimy_minus1 = gridDim.y - 1;

    const int index_n = index - cols;
    const int index_s = index + cols * BLOCK_SIZE;
    const int index_w = index - 1;
    const int index_e = index + BLOCK_SIZE;

    float n, w, e, s, jc, g2, l, num, den, qsqr, c;

    // shared memory allocation with padding to avoid bank conflicts
    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float temp_result[BLOCK_SIZE][BLOCK_SIZE + 1];

    __shared__ float north[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float south[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float east[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float west[BLOCK_SIZE][BLOCK_SIZE + 1];

    // load data to shared memory: north/south
    float north_val = J_cuda[index_n];
    float south_val = J_cuda[index_s];

    if (by == 0) {
        north_val = J_cuda[b_col + tx];
    } else if (by == gridDimy_minus1) {
        north_val = J_cuda[index_n];
        south_val = J_cuda[cols_block * gridDimy_minus1 +
                           b_col + cols * (BLOCK_SIZE - 1) + tx];
    }

    north[ty][tx] = north_val;
    south[ty][tx] = south_val;

    __syncthreads();

    // load data to shared memory: west/east
    float west_val = J_cuda[index_w];
    float east_val = J_cuda[index_e];

    if (bx == 0) {
        west_val = J_cuda[b_row * cols + cols * ty];
    } else if (bx == gridDimx_minus1) {
        west_val = J_cuda[index_w];
        east_val = J_cuda[b_row * cols +
                          BLOCK_SIZE * gridDimx_minus1 +
                          cols * ty + BLOCK_SIZE - 1];
    }

    west[ty][tx] = west_val;
    east[ty][tx] = east_val;

    __syncthreads();

    // central value
    jc = J_cuda[index];
    temp[ty][tx] = jc;

    __syncthreads();

    // compute neighbors
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
    } else if (ty == 0) { // n
        n = north[ty][tx] - jc;
        s = temp[ty + 1][tx] - jc;
        w = temp[ty][tx - 1] - jc;
        e = temp[ty][tx + 1] - jc;
    } else if (tx == BLOCK_SIZE - 1) { // e
        n = temp[ty - 1][tx] - jc;
        s = temp[ty + 1][tx] - jc;
        w = temp[ty][tx - 1] - jc;
        e = east[ty][tx] - jc;
    } else if (ty == BLOCK_SIZE - 1) { // s
        n = temp[ty - 1][tx] - jc;
        s = south[ty][tx] - jc;
        w = temp[ty][tx - 1] - jc;
        e = temp[ty][tx + 1] - jc;
    } else if (tx == 0) { // w
        n = temp[ty - 1][tx] - jc;
        s = temp[ty + 1][tx] - jc;
        w = west[ty][tx] - jc;
        e = temp[ty][tx + 1] - jc;
    } else { // interior
        n = temp[ty - 1][tx] - jc;
        s = temp[ty + 1][tx] - jc;
        w = temp[ty][tx - 1] - jc;
        e = temp[ty][tx + 1] - jc;
    }

    // core computations
    const float jc2 = jc * jc;
    const float n2 = n * n;
    const float s2 = s * s;
    const float w2 = w * w;
    const float e2 = e * e;

    g2 = (n2 + s2 + w2 + e2) / jc2;
    l  = (n + s + w + e) / jc;

    num = 0.5f * g2 - (0.0625f * l * l); // 1/16 = 0.0625
    den = 1.0f + 0.25f * l;
    qsqr = num / (den * den);

    den = (qsqr - q0sqr) / (q0sqr * (1.0f + q0sqr));
    c = 1.0f / (1.0f + den);

    // saturate diffusion coefficient
    c = fminf(fmaxf(c, 0.0f), 1.0f);
    temp_result[ty][tx] = c;

    __syncthreads();

    C_cuda[index] = temp_result[ty][tx];
    E_C[index]    = e;
    W_C[index]    = w;
    S_C[index]    = s;
    N_C[index]    = n;
}

__global__ void srad_cuda_2(float * __restrict__ E_C, float * __restrict__ W_C,
                            float * __restrict__ N_C, float * __restrict__ S_C,
                            float * __restrict__ J_cuda,
                            float * __restrict__ C_cuda,
                            int cols, int rows,
                            float lambda, float q0sqr) {

    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int b_col = BLOCK_SIZE * bx;
    const int b_row = BLOCK_SIZE * by;

    const int g_col = b_col + tx;
    const int g_row = b_row + ty;

    const int index = g_row * cols + g_col;

    const int cols_block = cols * BLOCK_SIZE;
    const int gridDimx_minus1 = gridDim.x - 1;
    const int gridDimy_minus1 = gridDim.y - 1;

    const int index_s = index + cols * BLOCK_SIZE;
    const int index_e = index + BLOCK_SIZE;

    float cc, cn, cs, ce, cw, d_sum;

    // shared memory allocation with padding to avoid bank conflicts
    __shared__ float south_c[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float east_c[BLOCK_SIZE][BLOCK_SIZE + 1];

    __shared__ float c_cuda_temp[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float c_cuda_result[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE + 1];

    // load data to shared memory
    float j_val = J_cuda[index];
    temp[ty][tx] = j_val;

    __syncthreads();

    float south_val = C_cuda[index_s];
    if (by == gridDimy_minus1) {
        south_val = C_cuda[cols_block * gridDimy_minus1 +
                           b_col + cols * (BLOCK_SIZE - 1) + tx];
    }
    south_c[ty][tx] = south_val;

    __syncthreads();

    float east_val = C_cuda[index_e];
    if (bx == gridDimx_minus1) {
        east_val = C_cuda[b_row * cols +
                          BLOCK_SIZE * gridDimx_minus1 +
                          cols * ty + BLOCK_SIZE - 1];
    }
    east_c[ty][tx] = east_val;

    __syncthreads();

    float c_val = C_cuda[index];
    c_cuda_temp[ty][tx] = c_val;

    __syncthreads();

    cc = c_cuda_temp[ty][tx];

    if (ty == BLOCK_SIZE - 1 && tx == BLOCK_SIZE - 1) { // se
        cn = cc;
        cs = south_c[ty][tx];
        cw = cc;
        ce = east_c[ty][tx];
    } else if (tx == BLOCK_SIZE - 1) { // e
        cn = cc;
        cs = c_cuda_temp[ty + 1][tx];
        cw = cc;
        ce = east_c[ty][tx];
    } else if (ty == BLOCK_SIZE - 1) { // s
        cn = cc;
        cs = south_c[ty][tx];
        cw = cc;
        ce = c_cuda_temp[ty][tx + 1];
    } else { // interior
        cn = cc;
        cs = c_cuda_temp[ty + 1][tx];
        cw = cc;
        ce = c_cuda_temp[ty][tx + 1];
    }

    // divergence (equ 58)
    d_sum = cn * N_C[index] + cs * S_C[index] +
            cw * W_C[index] + ce * E_C[index];

    // image update (equ 61)
    c_cuda_result[ty][tx] = temp[ty][tx] + 0.25f * lambda * d_sum;

    __syncthreads();

    J_cuda[index] = c_cuda_result[ty][tx];
}
