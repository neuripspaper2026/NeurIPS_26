#include "needle.h"
#include <stdio.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#define SDATA(index) CUT_BANK_CHECKER(sdata, index)

__device__ __forceinline__ int maximum(int a, int b, int c) {
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
    const int index    = base + tx + (cols + 1);
    const int index_n  = base + tx + 1;
    const int index_w  = base + cols;
    const int index_nw = base;

    __shared__ int temp[BLOCK_SIZE + 1][BLOCK_SIZE + 1];
    __shared__ int ref[BLOCK_SIZE][BLOCK_SIZE];

    // cooperative load of referrence with coalesced accesses
#pragma unroll
    for (int ty = 0; ty < BLOCK_SIZE; ty++) {
        ref[ty][tx] = referrence[index + cols * ty];
    }

    if (tx == 0) {
        temp[0][0] = matrix_cuda[index_nw];
    }

    __syncthreads();

    // load west and north borders
    temp[tx + 1][0] = matrix_cuda[index_w + cols * tx];
    temp[0][tx + 1] = matrix_cuda[index_n];

    __syncthreads();

    // first half anti-diagonals
#pragma unroll
    for (int m = 0; m < BLOCK_SIZE; m++) {
        if (tx <= m) {
            int t_index_x = tx + 1;
            int t_index_y = m - tx + 1;

            int up_left = temp[t_index_y - 1][t_index_x - 1];
            int left    = temp[t_index_y][t_index_x - 1];
            int up      = temp[t_index_y - 1][t_index_x];

            int score   = ref[t_index_y - 1][t_index_x - 1];

            temp[t_index_y][t_index_x] =
                maximum(up_left + score,
                        left - penalty,
                        up - penalty);
        }
        __syncthreads();
    }

    // second half anti-diagonals
#pragma unroll
    for (int m = BLOCK_SIZE - 2; m >= 0; m--) {
        if (tx <= m) {
            int t_index_x = tx + BLOCK_SIZE - m;
            int t_index_y = BLOCK_SIZE - tx;

            int up_left = temp[t_index_y - 1][t_index_x - 1];
            int left    = temp[t_index_y][t_index_x - 1];
            int up      = temp[t_index_y - 1][t_index_x];

            int score   = ref[t_index_y - 1][t_index_x - 1];

            temp[t_index_y][t_index_x] =
                maximum(up_left + score,
                        left - penalty,
                        up - penalty);
        }
        __syncthreads();
    }

    // store back results, coalesced by tx
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
    const int index    = base + tx + (cols + 1);
    const int index_n  = base + tx + 1;
    const int index_w  = base + cols;
    const int index_nw = base;

    __shared__ int temp[BLOCK_SIZE + 1][BLOCK_SIZE + 1];
    __shared__ int ref[BLOCK_SIZE][BLOCK_SIZE];

    // cooperative load of referrence with coalesced accesses
#pragma unroll
    for (int ty = 0; ty < BLOCK_SIZE; ty++) {
        ref[ty][tx] = referrence[index + cols * ty];
    }

    if (tx == 0) {
        temp[0][0] = matrix_cuda[index_nw];
    }

    __syncthreads();

    // load west and north borders
    temp[tx + 1][0] = matrix_cuda[index_w + cols * tx];
    temp[0][tx + 1] = matrix_cuda[index_n];

    __syncthreads();

    // first half anti-diagonals
#pragma unroll
    for (int m = 0; m < BLOCK_SIZE; m++) {
        if (tx <= m) {
            int t_index_x = tx + 1;
            int t_index_y = m - tx + 1;

            int up_left = temp[t_index_y - 1][t_index_x - 1];
            int left    = temp[t_index_y][t_index_x - 1];
            int up      = temp[t_index_y - 1][t_index_x];

            int score   = ref[t_index_y - 1][t_index_x - 1];

            temp[t_index_y][t_index_x] =
                maximum(up_left + score,
                        left - penalty,
                        up - penalty);
        }
        __syncthreads();
    }

    // second half anti-diagonals
#pragma unroll
    for (int m = BLOCK_SIZE - 2; m >= 0; m--) {
        if (tx <= m) {
            int t_index_x = tx + BLOCK_SIZE - m;
            int t_index_y = BLOCK_SIZE - tx;

            int up_left = temp[t_index_y - 1][t_index_x - 1];
            int left    = temp[t_index_y][t_index_x - 1];
            int up      = temp[t_index_y - 1][t_index_x];

            int score   = ref[t_index_y - 1][t_index_x - 1];

            temp[t_index_y][t_index_x] =
                maximum(up_left + score,
                        left - penalty,
                        up - penalty);
        }
        __syncthreads();
    }

    // store back results, coalesced by tx
#pragma unroll
    for (int ty = 0; ty < BLOCK_SIZE; ty++) {
        matrix_cuda[index + ty * cols] = temp[ty + 1][tx + 1];
    }
}
