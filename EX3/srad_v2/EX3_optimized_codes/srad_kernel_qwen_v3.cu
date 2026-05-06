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

    // Load center data
    temp[ty][tx] = J_cuda[index];
    __syncthreads();

    jc = temp[ty][tx];

    // Compute gradients with optimized boundary handling
    // Use predicated loads to avoid branching
    bool is_north_edge = (ty == 0);
    bool is_south_edge = (ty == BLOCK_SIZE - 1);
    bool is_west_edge = (tx == 0);
    bool is_east_edge = (tx == BLOCK_SIZE - 1);

    n = __ldg(&((is_north_edge) ? north[ty][tx] : temp[ty - 1][tx])) - jc;
    s = __ldg(&((is_south_edge) ? south[ty][tx] : temp[ty + 1][tx])) - jc;
    w = __ldg(&((is_west_edge) ? west[ty][tx] : temp[ty][tx - 1])) - jc;
    e = __ldg(&((is_east_edge) ? east[ty][tx] : temp[ty][tx + 1])) - jc;

    // Compute diffusion parameters
    g2 = (n * n + s * s + w * w + e * e) / (jc * jc);
    l = (n + s + w + e) / jc;

    num = (0.5f * g2) - ((1.0f / 16.0f) * (l * l));
    den = 1.0f + (0.25f * l);
    qsqr = num / (den * den);

    // diffusion coefficient (equ 33)
    den = (qsqr - q0sqr) / (q0sqr * (1.0f + q0sqr));
    c = 1.0f / (1.0f + den);

    // saturate diffusion coefficient with fminf/fmaxf for better performance
    c = fminf(fmaxf(c, 0.0f), 1.0f);
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

    // Load C_cuda data with boundary handling
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

    // Compute diffusion coefficients with optimized boundary handling
    bool is_south_edge = (ty == BLOCK_SIZE - 1);
    bool is_east_edge = (tx == BLOCK_SIZE - 1);

    cn = cc;
    cs = (is_south_edge) ? south_c[ty][tx] : c_cuda_temp[ty + 1][tx];
    cw = cc;
    ce = (is_east_edge) ? east_c[ty][tx] : c_cuda_temp[ty][tx + 1];

    // divergence (equ 58) - use register reuse
    d_sum = cn * N_C[index] + cs * S_C[index] + cw * W_C[index] + ce * E_C[index];

    // image update (equ 61)
    c_cuda_result[ty][tx] = temp[ty][tx] + 0.25f * lambda * d_sum;

    __syncthreads();

    J_cuda[index] = c_cuda_result[ty][tx];
}
