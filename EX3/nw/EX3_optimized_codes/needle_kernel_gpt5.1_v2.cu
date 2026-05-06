#include "needle.h"
#include <stdio.h>

#define SDATA(index) CUT_BANK_CHECKER(sdata, index)

// Use ternary operators to encourage inlining and avoid branch chains.
// Keep __host__ for compatibility with existing code.
__device__ __host__ __forceinline__ int maximum(int a, int b, int c) {
    int k = (a <= b) ? b : a;
    return (k <= c) ? c : k;
}

// Optimize shared memory layout to avoid bank conflicts on A100 (32 banks).
// Padding second dimension by +1 changes stride and reduces conflicts.
__global__ void needle_cuda_shared_1(int * __restrict__ referrence,
                                     int * __restrict__ matrix_cuda,
                                     int cols, int penalty, int i,
                                     int block_width) {
    const int bx = blockIdx.x;
    const int tx = threadIdx.x;

    const int b_index_x = bx;
    const int b_index_y = i - 1 - bx;

    const int base = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x;
    const int index   = base + tx + (cols + 1);
    const int index_n = base + tx + 1;
    const int index_w = base + cols;
    const int index_nw = base;

    // Pad second dimension to reduce shared-memory bank conflicts.
    __shared__ int temp[BLOCK_SIZE + 1][BLOCK_SIZE + 1 + 1];
    __shared__ int ref[BLOCK_SIZE][BLOCK_SIZE + 1];

    // Preload reference tile into shared memory (coalesced along columns).
#pragma unroll
    for (int ty = 0; ty < BLOCK_SIZE; ty++) {
        ref[ty][tx] = referrence[index + cols * ty];
    }

    __syncthreads();

    if (tx == 0) {
        temp[0][0] = matrix_cuda[index_nw];
    }

    // Coalesced loads along rows
    temp[tx + 1][0] = matrix_cuda[index_w + cols * tx];

    __syncthreads();

    temp[0][tx + 1] = matrix_cuda[index_n];

    __syncthreads();

    // First half-wavefront computation
#pragma unroll
    for (int m = 0; m < BLOCK_SIZE; m++) {
        if (tx <= m) {
            const int t_index_x = tx + 1;
            const int t_index_y = m - tx + 1;

            temp[t_index_y][t_index_x] =
                maximum(temp[t_index_y - 1][t_index_x - 1] +
                            ref[t_index_y - 1][t_index_x - 1],
                        temp[t_index_y][t_index_x - 1] - penalty,
                        temp[t_index_y - 1][t_index_x] - penalty);
        }
        __syncthreads();
    }

    // Second half-wavefront computation
#pragma unroll
    for (int m = BLOCK_SIZE - 2; m >= 0; m--) {
        if (tx <= m) {
            const int t_index_x = tx + BLOCK_SIZE - m;
            const int t_index_y = BLOCK_SIZE - tx;

            temp[t_index_y][t_index_x] =
                maximum(temp[t_index_y - 1][t_index_x - 1] +
                            ref[t_index_y - 1][t_index_x - 1],
                        temp[t_index_y][t_index_x - 1] - penalty,
                        temp[t_index_y - 1][t_index_x] - penalty);
        }
        __syncthreads();
    }

    // Store results back to global memory (coalesced along columns).
#pragma unroll
    for (int ty = 0; ty < BLOCK_SIZE; ty++) {
        matrix_cuda[index + ty * cols] = temp[ty + 1][tx + 1];
    }
}

// Same optimizations as needle_cuda_shared_1 applied here.
__global__ void needle_cuda_shared_2(int * __restrict__ referrence,
                                     int * __restrict__ matrix_cuda,
                                     int cols, int penalty, int i,
                                     int block_width) {

    const int bx = blockIdx.x;
    const int tx = threadIdx.x;

    const int b_index_x = bx + block_width - i;
    const int b_index_y = block_width - bx - 1;

    const int base = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x;
    const int index   = base + tx + (cols + 1);
    const int index_n = base + tx + 1;
    const int index_w = base + cols;
    const int index_nw = base;

    __shared__ int temp[BLOCK_SIZE + 1][BLOCK_SIZE + 1 + 1];
    __shared__ int ref[BLOCK_SIZE][BLOCK_SIZE + 1];

    // Preload reference tile (coalesced).
#pragma unroll
    for (int ty = 0; ty < BLOCK_SIZE; ty++) {
        ref[ty][tx] = referrence[index + cols * ty];
    }

    __syncthreads();

    if (tx == 0) {
        temp[0][0] = matrix_cuda[index_nw];
    }

    temp[tx + 1][0] = matrix_cuda[index_w + cols * tx];

    __syncthreads();

    temp[0][tx + 1] = matrix_cuda[index_n];

    __syncthreads();

#pragma unroll
    for (int m = 0; m < BLOCK_SIZE; m++) {
        if (tx <= m) {
            const int t_index_x = tx + 1;
            const int t_index_y = m - tx + 1;

            temp[t_index_y][t_index_x] =
                maximum(temp[t_index_y - 1][t_index_x - 1] +
                            ref[t_index_y - 1][t_index_x - 1],
                        temp[t_index_y][t_index_x - 1] - penalty,
                        temp[t_index_y - 1][t_index_x] - penalty);
        }
        __syncthreads();
    }

#pragma unroll
    for (int m = BLOCK_SIZE - 2; m >= 0; m--) {
        if (tx <= m) {
            const int t_index_x = tx + BLOCK_SIZE - m;
            const int t_index_y = BLOCK_SIZE - tx;

            temp[t_index_y][t_index_x] =
                maximum(temp[t_index_y - 1][t_index_x - 1] +
                            ref[t_index_y - 1][t_index_x - 1],
                        temp[t_index_y][t_index_x - 1] - penalty,
                        temp[t_index_y - 1][t_index_x] - penalty);
        }
        __syncthreads();
    }

#pragma unroll
    for (int ty = 0; ty < BLOCK_SIZE; ty++) {
        matrix_cuda[index + ty * cols] = temp[ty + 1][tx + 1];
    }
}
