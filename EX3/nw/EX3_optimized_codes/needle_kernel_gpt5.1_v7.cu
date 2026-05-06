#include "needle.h"
#include <stdio.h>

#define SDATA(index) CUT_BANK_CHECKER(sdata, index)

// Use __forceinline__ to encourage inlining on A100
__device__ __host__ __forceinline__ int maximum(int a, int b, int c) {
    int k = (a <= b) ? b : a;
    return (k <= c) ? c : k;
}

__global__ void needle_cuda_shared_1(int * __restrict__ referrence,
                                     int * __restrict__ matrix_cuda,
                                     int cols, int penalty, int i,
                                     int block_width) {
    // Ensure BLOCK_SIZE is a compile-time constant and power-of-two where possible
    const int bx = blockIdx.x;
    const int tx = threadIdx.x;

    const int b_index_x = bx;
    const int b_index_y = i - 1 - bx;

    const int base = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x;
    const int index   = base + tx + (cols + 1);
    const int index_n = base + tx + 1;
    const int index_w = base + cols;
    const int index_nw = base;

    // Pad the leading dimension to avoid shared memory bank conflicts
    __shared__ int temp[BLOCK_SIZE + 1][BLOCK_SIZE + 1 + 1];
    __shared__ int ref[BLOCK_SIZE][BLOCK_SIZE + 1];

    // Coalesced load of reference block into shared memory
#pragma unroll
    for (int ty = 0; ty < BLOCK_SIZE; ty++) {
        ref[ty][tx] = __ldg(&referrence[index + cols * ty]);
    }

    __syncthreads();

    if (tx == 0) {
        temp[0][0] = matrix_cuda[index_nw];
    }

    // Coalesced load of west border into shared memory
    temp[tx + 1][0] = matrix_cuda[index_w + cols * tx];

    __syncthreads();

    // Coalesced load of north border into shared memory
    temp[0][tx + 1] = matrix_cuda[index_n];

    __syncthreads();

    // First wavefront
#pragma unroll
    for (int m = 0; m < BLOCK_SIZE; m++) {
        if (tx <= m) {
            const int t_index_x = tx + 1;
            const int t_index_y = m - tx + 1;

            const int up_left = temp[t_index_y - 1][t_index_x - 1];
            const int left    = temp[t_index_y][t_index_x - 1];
            const int up      = temp[t_index_y - 1][t_index_x];
            const int score   = ref[t_index_y - 1][t_index_x - 1];

            temp[t_index_y][t_index_x] =
                maximum(up_left + score,
                        left - penalty,
                        up - penalty);
        }

        __syncthreads();
    }

    // Second wavefront
#pragma unroll
    for (int m = BLOCK_SIZE - 2; m >= 0; m--) {
        if (tx <= m) {
            const int t_index_x = tx + BLOCK_SIZE - m;
            const int t_index_y = BLOCK_SIZE - tx;

            const int up_left = temp[t_index_y - 1][t_index_x - 1];
            const int left    = temp[t_index_y][t_index_x - 1];
            const int up      = temp[t_index_y - 1][t_index_x];
            const int score   = ref[t_index_y - 1][t_index_x - 1];

            temp[t_index_y][t_index_x] =
                maximum(up_left + score,
                        left - penalty,
                        up - penalty);
        }

        __syncthreads();
    }

    // Coalesced store back to global memory
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

    __shared__ int temp[BLOCK_SIZE + 1][BLOCK_SIZE + 1 + 1];
    __shared__ int ref[BLOCK_SIZE][BLOCK_SIZE + 1];

    // Coalesced load of reference block into shared memory
#pragma unroll
    for (int ty = 0; ty < BLOCK_SIZE; ty++) {
        ref[ty][tx] = __ldg(&referrence[index + cols * ty]);
    }

    __syncthreads();

    if (tx == 0) {
        temp[0][0] = matrix_cuda[index_nw];
    }

    // Coalesced load of west border into shared memory
    temp[tx + 1][0] = matrix_cuda[index_w + cols * tx];

    __syncthreads();

    // Coalesced load of north border into shared memory
    temp[0][tx + 1] = matrix_cuda[index_n];

    __syncthreads();

    // First wavefront
#pragma unroll
    for (int m = 0; m < BLOCK_SIZE; m++) {

        if (tx <= m) {

            const int t_index_x = tx + 1;
            const int t_index_y = m - tx + 1;

            const int up_left = temp[t_index_y - 1][t_index_x - 1];
            const int left    = temp[t_index_y][t_index_x - 1];
            const int up      = temp[t_index_y - 1][t_index_x];
            const int score   = ref[t_index_y - 1][t_index_x - 1];

            temp[t_index_y][t_index_x] =
                maximum(up_left + score,
                        left - penalty,
                        up - penalty);
        }

        __syncthreads();
    }

    // Second wavefront
#pragma unroll
    for (int m = BLOCK_SIZE - 2; m >= 0; m--) {

        if (tx <= m) {

            const int t_index_x = tx + BLOCK_SIZE - m;
            const int t_index_y = BLOCK_SIZE - tx;

            const int up_left = temp[t_index_y - 1][t_index_x - 1];
            const int left    = temp[t_index_y][t_index_x - 1];
            const int up      = temp[t_index_y - 1][t_index_x];
            const int score   = ref[t_index_y - 1][t_index_x - 1];

            temp[t_index_y][t_index_x] =
                maximum(up_left + score,
                        left - penalty,
                        up - penalty);
        }

        __syncthreads();
    }

    // Coalesced store back to global memory
#pragma unroll
    for (int ty = 0; ty < BLOCK_SIZE; ty++) {
        matrix_cuda[index + ty * cols] = temp[ty + 1][tx + 1];
    }
}
