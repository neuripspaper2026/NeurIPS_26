#ifndef _BACKPROP_CUDA_KERNEL_H_
#define _BACKPROP_CUDA_KERNEL_H_

#include <stdio.h>
#include "../backprop.h"
#include "math.h"
#include "cuda.h"

__global__ void bpnn_layerforward_CUDA(float * __restrict__ input_cuda,
                                       float * __restrict__ output_hidden_cuda,
                                       float * __restrict__ input_hidden_cuda,
                                       float * __restrict__ hidden_partial_sum,
                                       int in,
                                       int hid) {
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int stride = hid + 1;

    // Precompute bases to reduce integer arithmetic in inner paths
    const int base_weight_row = stride * HEIGHT * by + stride * ty + 1 + stride;
    const int index = base_weight_row + tx;

    const int index_in = HEIGHT * by + ty + 1;

    __shared__ float input_node[HEIGHT];
    __shared__ float weight_matrix[HEIGHT][WIDTH];

    // Load input once per row in a coalesced way across warps
    if (tx == 0) {
        input_node[ty] = input_cuda[index_in];
    }

    __syncthreads();

    // Coalesced read of weight matrix from global memory
    weight_matrix[ty][tx] = input_hidden_cuda[index];

    __syncthreads();

    // Fused multiply: each thread computes its product
    float val = weight_matrix[ty][tx] * input_node[ty];
    weight_matrix[ty][tx] = val;

    __syncthreads();

    // Parallel reduction along ty dimension using shared memory
    // Assumes HEIGHT is a power of two (as in original code)
    for (int power_two = 2; power_two <= HEIGHT; power_two <<= 1) {
        if ((ty & (power_two - 1)) == 0) {
            int other = ty + (power_two >> 1);
#pragma unroll
            if (other < HEIGHT) {
                weight_matrix[ty][tx] += weight_matrix[other][tx];
            }
        }
        __syncthreads();
    }

    // Write back partial result for each (row, column)
    input_hidden_cuda[index] = weight_matrix[ty][tx];

    __syncthreads();

    // Store block-level partial sum used by later layers
    if (tx == 0) {
        hidden_partial_sum[by * hid + ty] = weight_matrix[0][ty];
    }
}


__global__ void bpnn_adjust_weights_cuda(float * __restrict__ delta,
                                         int hid,
                                         float * __restrict__ ly,
                                         int in,
                                         float * __restrict__ w,
                                         float * __restrict__ oldw) {

    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int stride = hid + 1;

    const int base_weight_row = stride * HEIGHT * by + stride * ty + 1 + stride;
    const int index = base_weight_row + tx;

    const int index_y = HEIGHT * by + ty + 1;
    const int index_x = tx + 1;

    const float d = delta[index_x];
    const float y = ly[index_y];

    const float grad = ETA * d * y;
    const float prev = oldw[index];
    const float update = grad + MOMENTUM * prev;

    w[index] += update;
    oldw[index] = update;

    __syncthreads();

    if (ty == 0 && by == 0) {
        const float d_root = delta[index_x];
        const float prev_root = oldw[index_x];
        const float update_root = ETA * d_root + MOMENTUM * prev_root;

        w[index_x] += update_root;
        oldw[index_x] = update_root;
    }
}
#endif
