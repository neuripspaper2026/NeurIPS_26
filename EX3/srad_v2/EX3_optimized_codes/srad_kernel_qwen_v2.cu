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

    // Load halo data with coalesced access patterns
    // North and South halo
    north[ty][tx] = J_cuda[index_n];
    south[ty][tx] = J_cuda[index_s];
    
    // Handle boundary conditions for north/south
    if (ty == 0 && by == 0) {
        north[0][tx] = J_cuda[BLOCK_SIZE * bx + tx];
    }
    if (ty == BLOCK_SIZE - 1 && by == gridDim.y - 1) {
        south[ty][tx] = J_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) +
                               BLOCK_SIZE * bx + cols * (BLOCK_SIZE - 1) + tx];
    }
    __syncthreads();

    // West and East halo
    west[ty][tx] = J_cuda[index_w];
    east[ty][tx] = J_cuda[index_e];

    // Handle boundary conditions for west/east
    if (tx == 0 && bx == 0) {
        west[ty][0] = J_cuda[cols * BLOCK_SIZE * by + cols * ty];
    }
    if (tx == BLOCK_SIZE - 1 && bx == gridDim.x - 1) {
        east[ty][tx] =
            J_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * (gridDim.x - 1) +
                   cols * ty + BLOCK_SIZE - 1];
    }

    __syncthreads();

    // Load center data
    temp[ty][tx] = J_cuda[index];
    __syncthreads();

    jc = temp[ty][tx];

    // Compute gradients with optimized boundary handling
    // Use predicated loads to avoid divergent branches
    bool is_north_edge = (ty == 0);
    bool is_south_edge = (ty == BLOCK_SIZE - 1);
    bool is_west_edge = (tx == 0);
    bool is_east_edge = (tx == BLOCK_SIZE - 1);

    n = __fsub_rn(is_north_edge ? north[ty][tx] : temp[ty - 1][tx], jc);
    s = __fsub_rn(is_south_edge ? south[ty][tx] : temp[ty + 1][tx], jc);
    w = __fsub_rn(is_west_edge ? west[ty][tx] : temp[ty][tx - 1], jc);
    e = __fsub_rn(is_east_edge ? east[ty][tx] : temp[ty][tx + 1], jc);

    // Compute diffusion with FMAs for better precision and performance
    float n2 = __fmul_rn(n, n);
    float s2 = __fmul_rn(s, s);
    float w2 = __fmul_rn(w, w);
    float e2 = __fmul_rn(e, e);
    
    g2 = __fdividef(__fadd_rn(__fadd_rn(__fadd_rn(n2, s2), w2), e2), __fmul_rn(jc, jc));

    l = __fdividef(__fadd_rn(__fadd_rn(__fadd_rn(n, s), w), e), jc);

    num = __fsub_rn(__fmul_rn(0.5f, g2), __fmul_rn(0.0625f, __fmul_rn(l, l)));
    den = __fadd_rn(1.0f, __fmul_rn(0.25f, l));
    qsqr = __fdividef(num, __fmul_rn(den, den));

    // diffusion coefficient (equ 33)
    den = __fdividef(__fsub_rn(qsqr, q0sqr), __fmul_rn(q0sqr, __fadd_rn(1.0f, q0sqr)));
    c = __fdividef(1.0f, __fadd_rn(1.0f, den));

    // saturate diffusion coefficient using min/max for better performance
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

    // Load data with coalesced access
    temp[ty][tx] = J_cuda[index];
    __syncthreads();

    // Load coefficient data
    south_c[ty][tx] = C_cuda[index_s];
    east_c[ty][tx] = C_cuda[index_e];

    // Handle boundary conditions
    if (ty == BLOCK_SIZE - 1 && by == gridDim.y - 1) {
        south_c[ty][tx] =
            C_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) + BLOCK_SIZE * bx +
                   cols * (BLOCK_SIZE - 1) + tx];
    }
    if (tx == BLOCK_SIZE - 1 && bx == gridDim.x - 1) {
        east_c[ty][tx] =
            C_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * (gridDim.x - 1) +
                   cols * ty + BLOCK_SIZE - 1];
    }
    __syncthreads();

    c_cuda_temp[ty][tx] = C_cuda[index];
    __syncthreads();

    cc = c_cuda_temp[ty][tx];

    // Compute coefficients with reduced branching
    bool is_south_edge = (ty == BLOCK_SIZE - 1);
    bool is_east_edge = (tx == BLOCK_SIZE - 1);

    cs = is_south_edge ? south_c[ty][tx] : c_cuda_temp[ty + 1][tx];
    ce = is_east_edge ? east_c[ty][tx] : c_cuda_temp[ty][tx + 1];
    cn = cc;
    cw = cc;

    // divergence (equ 58) - use FMAs for better performance
    d_sum = __fmaf_rn(cn, N_C[index], __fmaf_rn(cs, S_C[index], 
            __fmaf_rn(cw, W_C[index], __fmul_rn(ce, E_C[index]))));

    // image update (equ 61) - use FMA
    c_cuda_result[ty][tx] = __fmaf_rn(0.25f, __fmul_rn(lambda, d_sum), temp[ty][tx]);

    __syncthreads();

    J
