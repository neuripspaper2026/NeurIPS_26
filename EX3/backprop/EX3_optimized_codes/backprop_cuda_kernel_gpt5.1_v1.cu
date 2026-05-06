#ifndef _BACKPROP_CUDA_KERNEL_H_
#define _BACKPROP_CUDA_KERNEL_H_

#include <stdio.h>
#include "../backprop.h"
#include "math.h"
#include "cuda.h"

__global__ void bpnn_layerforward_CUDA(float *input_cuda,
                                       float *output_hidden_cuda,
                                       float *input_hidden_cuda,
                                       float *hidden_partial_sum, int in,
                                       int hid) {
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Precompute stride to avoid repeated integer multiplications
    int stride = (hid + 1) * HEIGHT;
    int index = stride * by + (hid + 1) * ty + tx + 1 + (hid + 1);
    int index_in = HEIGHT * by + ty + 1;

    __shared__ float input_node[HEIGHT];
    __shared__ float weight_matrix[HEIGHT][WIDTH];

    // Load input once per row using the first thread in x-dimension
    if (tx == 0) {
        input_node[ty] = __ldg(&input_cuda[index_in]);
    }

    __syncthreads();

    // Coalesced load of weights
    weight_matrix[ty][tx] = __ldg(&input_hidden_cuda[index]);

    __syncthreads();

    // Multiply by broadcasted input
    float val = weight_matrix[ty][tx] * input_node[ty];
    weight_matrix[ty][tx] = val;

    __syncthreads();

    // Tree reduction in shared memory across rows (ty dimension)
    // Assumes HEIGHT is a power of two as used in original code
    for (int power_two = 2; power_two <= HEIGHT; power_two <<= 1) {
        if ((ty & (power_two - 1)) == 0) {
            weight_matrix[ty][tx] += weight_matrix[ty + (power_two >> 1)][tx];
        }
        __syncthreads();
    }

    // Write back partial results
    input_hidden_cuda[index] = weight_matrix[ty][tx];

    __syncthreads();

    // Store block-level partial sums
    if (tx == 0) {
        hidden_partial_sum[by * hid + ty] = weight_matrix[0][ty];
    }

    // Prevent compiler from optimizing away unused parameter
    (void)output_hidden_cuda;
    (void)in;
}

__global__ void bpnn_adjust_weights_cuda(float *delta, int hid, float *ly,
                                         int in, float *w, float *oldw) {
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int stride = (hid + 1) * HEIGHT;
    int index = stride * by + (hid + 1) * ty + tx + 1 + (hid + 1);
    int index_y = HEIGHT * by + ty + 1;
    int index_x = tx + 1;

    // Load frequently used values into registers
    float delta_x = __ldg(&delta[index_x]);
    float ly_y    = __ldg(&ly[index_y]);
    float oldw_val = oldw[index];

    float update = ETA * delta_x * ly_y + MOMENTUM * oldw_val;

    w[index]    = w[index] + update;
    oldw[index] = update;

    __syncthreads();

    if (ty == 0 && by == 0) {
        float oldw_x = oldw[index_x];
        float update_x = ETA * delta_x + MOMENTUM * oldw_x;
        w[index_x]    = w[index_x] + update_x;
        oldw[index_x] = update_x;
    }

    // Prevent compiler from optimizing away unused parameter
    (void)in;
}
#endif
