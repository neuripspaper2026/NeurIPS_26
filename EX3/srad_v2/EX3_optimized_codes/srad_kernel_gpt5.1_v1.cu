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

    const int col_base = BLOCK_SIZE * bx;
    const int row_base = BLOCK_SIZE * by;
    const int g_col = col_base + tx;
    const int g_row = row_base + ty;
    const int index = g_row * cols + g_col;

    const int index_n = index - cols;
    const int index_s = index + cols * BLOCK_SIZE;
    const int index_w = index - 1;
    const int index_e = index + BLOCK_SIZE;

    float n, w, e, s, jc, g2, l, num, den, qsqr, c;

    // shared memory allocation (padded second dimension to reduce bank conflicts)
    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float temp_result[BLOCK_SIZE][BLOCK_SIZE + 1];

    __shared__ float north[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float south[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float east[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float west[BLOCK_SIZE][BLOCK_SIZE + 1];

    // precompute boundary flags
    const bool is_first_block_y = (by == 0);
    const bool is_last_block_y  = (by == gridDim.y - 1);
    const bool is_first_block_x = (bx == 0);
    const bool is_last_block_x  = (bx == gridDim.x - 1);

    // load data to shared memory (north / south)
    north[ty][tx] = J_cuda[index_n];
    south[ty][tx] = J_cuda[index_s];

    if (is_first_block_y) {
        north[ty][tx] = J_cuda[col_base + tx];
    } else if (is_last_block_y) {
        south[ty][tx] = J_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) +
                               col_base + cols * (BLOCK_SIZE - 1) + tx];
    }

    __syncthreads();

    // load data to shared memory (west / east)
    west[ty][tx] = J_cuda[index_w];
    east[ty][tx] = J_cuda[index_e];

    if (is_first_block_x) {
        west[ty][tx] = J_cuda[row_base * cols + cols * ty];
    } else if (is_last_block_x) {
        east[ty][tx] =
            J_cuda[row_base * cols + BLOCK_SIZE * (gridDim.x - 1) +
                   cols * ty + BLOCK_SIZE - 1];
    }

    __syncthreads();

    temp[ty][tx] = J_cuda[index];

    __syncthreads();

    jc = temp[ty][tx];

    // compute directional differences with minimized branching
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
    } else { // inner elements
        n = temp[ty - 1][tx] - jc;
        s = temp[ty + 1][tx] - jc;
        w = temp[ty][tx - 1] - jc;
        e = temp[ty][tx + 1] - jc;
    }

    const float jc2 = jc * jc;
    g2 = (n * n + s * s + w * w + e * e) / jc2;
    l = (n + s + w + e) / jc;

    num = 0.5f * g2 - (0.0625f * (l * l));
    den = 1.0f + (0.25f * l);
    qsqr = num / (den * den);

    den = (qsqr - q0sqr) / (q0sqr * (1.0f + q0sqr));
    c = 1.0f / (1.0f + den);

    // saturate diffusion coefficient with branchless clamps using fminf/fmaxf
    c = fminf(fmaxf(c, 0.0f), 1.0f);
    temp_result[ty][tx] = c;

    __syncthreads();

    C_cuda[index] = temp_result[ty][tx];
    E_C[index] = e;
    W_C[index] = w;
    S_C[index] = s;
    N_C[index] = n;
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

    const int col_base = BLOCK_SIZE * bx;
    const int row_base = BLOCK_SIZE * by;
    const int g_col = col_base + tx;
    const int g_row = row_base + ty;
    const int index = g_row * cols + g_col;

    const int index_s = index + cols * BLOCK_SIZE;
    const int index_e = index + BLOCK_SIZE;

    float cc, cn, cs, ce, cw, d_sum;

    // shared memory allocation (padded to reduce bank conflicts)
    __shared__ float south_c[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float east_c[BLOCK_SIZE][BLOCK_SIZE + 1];

    __shared__ float c_cuda_temp[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float c_cuda_result[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE + 1];

    const bool is_last_block_y = (by == gridDim.y - 1);
    const bool is_last_block_x = (bx == gridDim.x - 1);

    // load data to shared memory
    temp[ty][tx] = J_cuda[index];

    __syncthreads();

    south_c[ty][tx] = C_cuda[index_s];

    if (is_last_block_y) {
        south_c[ty][tx] =
            C_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) + col_base +
                   cols * (BLOCK_SIZE - 1) + tx];
    }
    __syncthreads();

    east_c[ty][tx] = C_cuda[index_e];

    if (is_last_block_x) {
        east_c[ty][tx] =
            C_cuda[row_base * cols + BLOCK_SIZE * (gridDim.x - 1) +
                   cols * ty + BLOCK_SIZE - 1];
    }

    __syncthreads();

    c_cuda_temp[ty][tx] = C_cuda[index];

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
    } else { // inner elements
        cn = cc;
        cs = c_cuda_temp[ty + 1][tx];
        cw = cc;
        ce = c_cuda_temp[ty][tx + 1];
    }

    d_sum = cn * N_C[index] + cs * S_C[index] +
            cw * W_C[index] + ce * E_C[index];

    c_cuda_result[ty][tx] = temp[ty][tx] + 0.25f * lambda * d_sum;

    __syncthreads();

    J_cuda[index] = c_cuda_result[ty][tx];
}
