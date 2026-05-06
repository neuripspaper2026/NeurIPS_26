#include "needle.h"
#include <stdio.h>

#define SDATA(index) CUT_BANK_CHECKER(sdata, index)

// Use device-only __forceinline__ for faster execution on GPU;
// host code can use a separate inline version if needed.
__device__ __forceinline__ int maximum(int a, int b, int c) {
    int k = (a <= b) ? b : a;
    return (k <= c) ? c : k;
}

// Host-side inline version to preserve __host__ functionality if required elsewhere.
__host__ __forceinline__ int maximum_host(int a, int b, int c) {
    int k = (a <= b) ? b : a;
    return (k <= c) ? c : k;
}

__global__ void needle_cuda_shared_1(int *__restrict__ referrence,
                                     int *__restrict__ matrix_cuda,
                                     int cols, int penalty, int i,
                                     int block_width) {
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int b_index_x = bx;
    int b_index_y = i - 1 - bx;

    int base = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x;
    int index    = base + tx + (cols + 1);
    int index_n  = base + tx + 1;
    int index_w  = base + cols;
    int index_nw = base;

    // Pad second dimension by +1 to avoid shared-memory bank conflicts
    __shared__ int temp[BLOCK_SIZE + 1][BLOCK_SIZE + 1 + 1];
    __shared__ int ref[BLOCK_SIZE][BLOCK_SIZE + 1];

    if (tx == 0) {
        temp[0][0] = matrix_cuda[index_nw];
    }

    // Coalesced load of referrence into shared memory
#pragma unroll
    for (int ty = 0; ty < BLOCK_SIZE; ty++) {
        ref[ty][tx] = referrence[index + cols * ty];
    }

    __syncthreads();

    // Coalesced load of west boundary into shared memory
#pragma unroll
    for (int offset = 0; offset < BLOCK_SIZE; offset += blockDim.x) {
        int ty = tx + offset;
        if (ty < BLOCK_SIZE) {
            temp[ty + 1][0] = matrix_cuda[index_w + cols * ty];
        }
    }

    __syncthreads();

    // Coalesced load of north boundary into shared memory
    temp[0][tx + 1] = matrix_cuda[index_n];

    __syncthreads();

#pragma unroll
    for (int m = 0; m < BLOCK_SIZE; m++) {
        if (tx <= m) {
            int t_index_x = tx + 1;
            int t_index_y = m - tx + 1;

            int v1 = temp[t_index_y - 1][t_index_x - 1] +
                     ref[t_index_y - 1][t_index_x - 1];
            int v2 = temp[t_index_y][t_index_x - 1] - penalty;
            int v3 = temp[t_index_y - 1][t_index_x] - penalty;

            temp[t_index_y][t_index_x] = maximum(v1, v2, v3);
        }
        __syncthreads();
    }

#pragma unroll
    for (int m = BLOCK_SIZE - 2; m >= 0; m--) {
        if (tx <= m) {
            int t_index_x = tx + BLOCK_SIZE - m;
            int t_index_y = BLOCK_SIZE - tx;

            int v1 = temp[t_index_y - 1][t_index_x - 1] +
                     ref[t_index_y - 1][t_index_x - 1];
            int v2 = temp[t_index_y][t_index_x - 1] - penalty;
            int v3 = temp[t_index_y - 1][t_index_x] - penalty;

            temp[t_index_y][t_index_x] = maximum(v1, v2, v3);
        }
        __syncthreads();
    }

#pragma unroll
    for (int ty = 0; ty < BLOCK_SIZE; ty++) {
        matrix_cuda[index + ty * cols] = temp[ty + 1][tx + 1];
    }
}

__global__ void needle_cuda_shared_2(int *__restrict__ referrence,
                                     int *__restrict__ matrix_cuda,
                                     int cols, int penalty, int i,
                                     int block_width) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int b_index_x = bx + block_width - i;
    int b_index_y = block_width - bx - 1;

    int base = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x;
    int index    = base + tx + (cols + 1);
    int index_n  = base + tx + 1;
    int index_w  = base + cols;
    int index_nw = base;

    __shared__ int temp[BLOCK_SIZE + 1][BLOCK_SIZE + 1 + 1];
    __shared__ int ref[BLOCK_SIZE][BLOCK_SIZE + 1];

    // Coalesced load of referrence into shared memory
#pragma unroll
    for (int ty = 0; ty < BLOCK_SIZE; ty++) {
        ref[ty][tx] = referrence[index + cols * ty];
    }

    __syncthreads();

    if (tx == 0) {
        temp[0][0] = matrix_cuda[index_nw];
    }

    // Coalesced load of west boundary into shared memory
#pragma unroll
    for (int offset = 0; offset < BLOCK_SIZE; offset += blockDim.x) {
        int ty = tx + offset;
        if (ty < BLOCK_SIZE) {
            temp[ty + 1][0] = matrix_cuda[index_w + cols * ty];
        }
    }

    __syncthreads();

    // Coalesced load of north boundary into shared memory
    temp[0][tx + 1] = matrix_cuda[index_n];

    __syncthreads();

#pragma unroll
    for (int m = 0; m < BLOCK_SIZE; m++) {
        if (tx <= m) {
            int t_index_x = tx + 1;
            int t_index_y = m - tx + 1;

            int v1 = temp[t_index_y - 1][t_index_x - 1] +
                     ref[t_index_y - 1][t_index_x - 1];
            int v2 = temp[t_index_y][t_index_x - 1] - penalty;
            int v3 = temp[t_index_y - 1][t_index_x] - penalty;

            temp[t_index_y][t_index_x] = maximum(v1, v2, v3);
        }
        __syncthreads();
    }

#pragma unroll
    for (int m = BLOCK_SIZE - 2; m >= 0; m--) {
        if (tx <= m) {
            int t_index_x = tx + BLOCK_SIZE - m;
            int t_index_y = BLOCK_SIZE - tx;

            int v1 = temp[t_index_y - 1][t_index_x - 1] +
                     ref[t_index_y - 1][t_index_x - 1];
            int v2 = temp[t_index_y][t_index_x - 1] - penalty;
            int v3 = temp[t_index_y - 1][t_index_x] - penalty;

            temp[t_index_y][t_index_x] = maximum(v1, v2, v3);
        }
        __syncthreads();
    }

#pragma unroll
    for (int ty = 0; ty < BLOCK_SIZE; ty++) {
        matrix_cuda[index + ty * cols] = temp[ty + 1][tx + 1];
    }
}
