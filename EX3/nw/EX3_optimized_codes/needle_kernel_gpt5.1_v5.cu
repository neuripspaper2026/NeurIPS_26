#include "needle.h"
#include <stdio.h>

#define SDATA(index) CUT_BANK_CHECKER(sdata, index)

__device__ __forceinline__ int maximum(int a, int b, int c) {
    int k = (a <= b) ? b : a;
    return (k <= c) ? c : k;
}

__global__ void needle_cuda_shared_1(int * __restrict__ referrence, int * __restrict__ matrix_cuda,
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

    // load reference block, coalesced in x, loop in y
#pragma unroll
    for (int ty = 0; ty < BLOCK_SIZE; ty++) {
        ref[ty][tx] = referrence[index + cols * ty];
    }

    __syncthreads();

    if (tx == 0) {
        temp[0][0] = matrix_cuda[index_nw];
    }

    // first column of tile (south of NW)
#pragma unroll
    for (int ty = 0; ty < BLOCK_SIZE; ty += blockDim.x) {
        int row = ty + tx;
        if (row < BLOCK_SIZE) {
            temp[row + 1][0] = matrix_cuda[index_w + cols * row];
        }
    }

    __syncthreads();

    // first row of tile (east of NW)
    temp[0][tx + 1] = matrix_cuda[index_n];

    __syncthreads();

    // main NW->SE wavefront
#pragma unroll
    for (int m = 0; m < BLOCK_SIZE; m++) {
        if (tx <= m) {
            int t_index_x = tx + 1;
            int t_index_y = m - tx + 1;

            int v_nw = temp[t_index_y - 1][t_index_x - 1];
            int v_w  = temp[t_index_y][t_index_x - 1];
            int v_n  = temp[t_index_y - 1][t_index_x];

            int r    = ref[t_index_y - 1][t_index_x - 1];

            temp[t_index_y][t_index_x] =
                maximum(v_nw + r,
                        v_w  - penalty,
                        v_n  - penalty);
        }
        __syncthreads();
    }

    // main SE->NW wavefront
#pragma unroll
    for (int m = BLOCK_SIZE - 2; m >= 0; m--) {
        if (tx <= m) {
            int t_index_x = tx + BLOCK_SIZE - m;
            int t_index_y = BLOCK_SIZE - tx;

            int v_nw = temp[t_index_y - 1][t_index_x - 1];
            int v_w  = temp[t_index_y][t_index_x - 1];
            int v_n  = temp[t_index_y - 1][t_index_x];

            int r    = ref[t_index_y - 1][t_index_x - 1];

            temp[t_index_y][t_index_x] =
                maximum(v_nw + r,
                        v_w  - penalty,
                        v_n  - penalty);
        }
        __syncthreads();
    }

    // store result tile back to global memory
#pragma unroll
    for (int ty = 0; ty < BLOCK_SIZE; ty++) {
        matrix_cuda[index + ty * cols] = temp[ty + 1][tx + 1];
    }
}

__global__ void needle_cuda_shared_2(int * __restrict__ referrence, int * __restrict__ matrix_cuda,
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

    // load reference block, coalesced in x, loop in y
#pragma unroll
    for (int ty = 0; ty < BLOCK_SIZE; ty++) {
        ref[ty][tx] = referrence[index + cols * ty];
    }

    __syncthreads();

    if (tx == 0) {
        temp[0][0] = matrix_cuda[index_nw];
    }

    // first column of tile (south of NW)
#pragma unroll
    for (int ty = 0; ty < BLOCK_SIZE; ty += blockDim.x) {
        int row = ty + tx;
        if (row < BLOCK_SIZE) {
            temp[row + 1][0] = matrix_cuda[index_w + cols * row];
        }
    }

    __syncthreads();

    // first row of tile (east of NW)
    temp[0][tx + 1] = matrix_cuda[index_n];

    __syncthreads();

    // main NW->SE wavefront
#pragma unroll
    for (int m = 0; m < BLOCK_SIZE; m++) {
        if (tx <= m) {
            int t_index_x = tx + 1;
            int t_index_y = m - tx + 1;

            int v_nw = temp[t_index_y - 1][t_index_x - 1];
            int v_w  = temp[t_index_y][t_index_x - 1];
            int v_n  = temp[t_index_y - 1][t_index_x];

            int r    = ref[t_index_y - 1][t_index_x - 1];

            temp[t_index_y][t_index_x] =
                maximum(v_nw + r,
                        v_w  - penalty,
                        v_n  - penalty);
        }
        __syncthreads();
    }

    // main SE->NW wavefront
#pragma unroll
    for (int m = BLOCK_SIZE - 2; m >= 0; m--) {
        if (tx <= m) {
            int t_index_x = tx + BLOCK_SIZE - m;
            int t_index_y = BLOCK_SIZE - tx;

            int v_nw = temp[t_index_y - 1][t_index_x - 1];
            int v_w  = temp[t_index_y][t_index_x - 1];
            int v_n  = temp[t_index_y - 1][t_index_x];

            int r    = ref[t_index_y - 1][t_index_x - 1];

            temp[t_index_y][t_index_x] =
                maximum(v_nw + r,
                        v_w  - penalty,
                        v_n  - penalty);
        }
        __syncthreads();
    }

    // store result tile back to global memory
#pragma unroll
    for (int ty = 0; ty < BLOCK_SIZE; ty++) {
        matrix_cuda[index + ty * cols] = temp[ty + 1][tx + 1];
    }
}
