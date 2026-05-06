#ifndef _BACKPROP_CUDA_KERNEL_H_
#define _BACKPROP_CUDA_KERNEL_H_

#include <stdio.h>
#include "../backprop.h"
#include "math.h"
#include "cuda.h"

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Utility for warp-level reduction (sum) on a single float
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void bpnn_layerforward_CUDA(float *input_cuda,
                                       float *output_hidden_cuda,
                                       float *input_hidden_cuda,
                                       float *hidden_partial_sum, int in,
                                       int hid) {
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int index = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1);
    int index_in = HEIGHT * by + ty + 1;

    __shared__ float input_node[HEIGHT];
    // Transpose shared memory to improve write/read coalescing
    __shared__ float weight_matrix[WIDTH][HEIGHT];

    // Load input_node only once per row; broadcast via shared memory
    if (tx == 0) {
        input_node[ty] = input_cuda[index_in];
    }

    __syncthreads();

    // Use read-only cache for input_hidden_cuda
    float w_val = __ldg(&input_hidden_cuda[index]);
    float in_val = input_node[ty];
    float prod = w_val * in_val;

    // Store with transposed indices to favor coalesced access
    weight_matrix[tx][ty] = prod;

    __syncthreads();

    // Perform reduction over HEIGHT dimension using warp-level primitives
    // Assuming HEIGHT <= WARP_SIZE or multiple of warp size;
    // we let each (tx) column perform reduction across ty
    float sum = 0.0f;
    if (ty < HEIGHT) {
        sum = weight_matrix[tx][ty];
    }

    // Reduce within warp along ty dimension
    sum = warpReduceSum(sum);

    // Write back reduced value for this (tx, column) at ty == 0
    if (ty == 0) {
        // Use original index pattern; consolidate result into row 0
        int base_index = (hid + 1) * HEIGHT * by + (hid + 1) * 0 + tx + 1 + (hid + 1);
        input_hidden_cuda[base_index] = sum;
        // Keep compatibility with original storage for further usage
        weight_matrix[tx][0] = sum;
    }

    __syncthreads();

    // Maintain original output behavior: store per-thread partial sum
    // Map to transposed shared memory layout; avoid bank conflicts by using tx-major
    int write_index = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1);
    if (ty == 0) {
        input_hidden_cuda[write_index] = weight_matrix[tx][0];
    } else {
        // For other rows, keep original product for compatibility
        input_hidden_cuda[write_index] = weight_matrix[tx][ty];
    }

    __syncthreads();

    // Preserve original hidden_partial_sum behavior
    if (tx == 0) {
        hidden_partial_sum[by * hid + ty] = weight_matrix[0][ty];
    }
}

__global__ void bpnn_adjust_weights_cuda(float *delta, int hid, float *ly,
                                         int in, float *w, float *oldw) {

    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int index = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1);
    int index_y = HEIGHT * by + ty + 1;
    int index_x = tx + 1;

    // Use registers and read-only cache for inputs
    float delta_val = __ldg(&delta[index_x]);
    float ly_val    = __ldg(&ly[index_y]);
    float oldw_val  = oldw[index];

    float update = ETA * delta_val * ly_val + MOMENTUM * oldw_val;
    w[index]     = w[index] + update;
    oldw[index]  = update;

    __syncthreads();

    // Bias update (ty == 0 and by == 0)
    if (ty == 0 && by == 0) {
        float bias_oldw = oldw[index_x];
        float bias_update = ETA * delta_val + MOMENTUM * bias_oldw;
        w[index_x]    = w[index_x] + bias_update;
        oldw[index_x] = bias_update;
    }
}
#endif
