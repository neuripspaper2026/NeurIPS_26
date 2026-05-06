#include "needle.h"
#include <stdio.h>

#define SDATA(index) CUT_BANK_CHECKER(sdata, index)

__device__ __host__ __forceinline__ int maximum(int a, int b, int c) {
    int k = (a > b) ? a : b;
    return (k > c) ? k : c;
}

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

    __shared__ int temp[BLOCK_SIZE + 1][BLOCK_SIZE + 1];
    __shared__ int ref[BLOCK_SIZE][BLOCK_SIZE];

    int lref[BLOCK_SIZE];

#pragma unroll
    for (int ty = 0; ty < BLOCK_SIZE; ty++) {
        lref[ty] = referrence[index + cols * ty];
    }

#pragma unroll
    for (int ty = 0; ty < BLOCK_SIZE; ty++) {
        ref[ty][tx] = lref[ty];
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

            const int up_left = temp[t_index_y - 1][t_index_x - 1] +
                                ref[t_index_y - 1][t_index_x - 1];
            const int left    = temp[t_index_y][t_index_x - 1] - penalty;
            const int up      = temp[t_index_y - 1][t_index_x] - penalty;

            temp[t_index_y][t_index_x] = maximum(up_left, left, up);
        }

        __syncthreads();
    }

#pragma unroll
    for (int m = BLOCK_SIZE - 2; m >= 0; m--) {
        if (tx <= m) {
            const int t_index_x = tx + BLOCK_SIZE - m;
            const int t_index_y = BLOCK_SIZE - tx;

            const int up_left = temp[t_index_y - 1][t_index_x - 1] +
                                ref[t_index_y - 1][t_index_x - 1];
            const int left    = temp[t_index_y][t_index_x - 1] - penalty;
            const int up      = temp[t_index_y - 1][t_index_x] - penalty;

            temp[t_index_y][t_index_x] = maximum(up_left, left, up);
        }

        __syncthreads();
    }

#pragma unroll
    for (int ty = 0; ty < BLOCK_SIZE; ty++) {
        matrix_cuda[index + ty * cols] = temp[ty + 1][tx + 1];
    }
}

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

    __shared__ int temp[BLOCK_SIZE + 1][BLOCK_SIZE + 1];
    __shared__ int ref[BLOCK_SIZE][BLOCK_SIZE];

    int lref[BLOCK_SIZE];

#pragma unroll
    for (int ty = 0; ty < BLOCK_SIZE; ty++) {
        lref[ty] = referrence[index + cols * ty];
    }

#pragma unroll
    for (int ty = 0; ty < BLOCK_SIZE; ty++) {
        ref[ty][tx] = lref[ty];
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

            const int up_left = temp[t_index_y - 1][t_index_x - 1] +
                                ref[t_index_y - 1][t_index_x - 1];
            const int left    = temp[t_index_y][t_index_x - 1] - penalty;
            const int up      = temp[t_index_y - 1][t_index_x] - penalty;

            temp[t_index_y][t_index_x] = maximum(up_left, left, up);
        }

        __syncthreads();
    }

#pragma unroll
    for (int m = BLOCK_SIZE - 2; m >= 0; m--) {

        if (tx <= m) {
            const int t_index_x = tx + BLOCK_SIZE - m;
            const int t_index_y = BLOCK_SIZE - tx;

            const int up_left = temp[t_index_y - 1][t_index_x - 1] +
                                ref[t_index_y - 1][t_index_x - 1];
            const int left    = temp[t_index_y][t_index_x - 1] - penalty;
            const int up      = temp[t_index_y - 1][t_index_x] - penalty;

            temp[t_index_y][t_index_x] = maximum(up_left, left, up);
        }

        __syncthreads();
    }

#pragma unroll
    for (int ty = 0; ty < BLOCK_SIZE; ty++) {
        matrix_cuda[index + ty * cols] = temp[ty + 1][tx + 1];
    }
}
