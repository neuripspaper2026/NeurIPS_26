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

    int index = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1);
    int index_in = HEIGHT * by + ty + 1;

    // Use proper shared memory layout for coalesced access
    __shared__ float input_node[HEIGHT];
    __shared__ float weight_matrix[HEIGHT][WIDTH];

    // Coalesced load of input data
    if (tx == 0)
        input_node[ty] = input_cuda[index_in];

    __syncthreads();

    // Coalesced load of weights
    weight_matrix[ty][tx] = input_hidden_cuda[index];

    __syncthreads();

    // Vectorized multiplication
    weight_matrix[ty][tx] = weight_matrix[ty][tx] * input_node[ty];

    __syncthreads();

    // Optimized reduction with better memory access pattern
    for (int power_two = 2; power_two <= HEIGHT; power_two *= 2) {
        int jump = power_two / 2;
        if (ty % power_two == 0 && (ty + jump) < HEIGHT)
            weight_matrix[ty][tx] += weight_matrix[ty + jump][tx];
        __syncthreads();
    }

    // Store result back
    input_hidden_cuda[index] = weight_matrix[ty][tx];

    __syncthreads();

    // Coalesced write to partial sum
    if (tx == 0 && ty < hid) {
        hidden_partial_sum[by * hid + ty] = weight_matrix[ty][0];
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

    // Use registers to reduce global memory accesses
    float delta_val = delta[index_x];
    float ly_val = ly[index_y];
    float w_val = w[index];
    float oldw_val = oldw[index];

    // Combined computation
    float gradient = ETA * delta_val * ly_val;
    float momentum_term = MOMENTUM * oldw_val;
    float weight_update = gradient + momentum_term;

    // Update weights
    w[index] = w_val + weight_update;
    oldw[index] = weight_update;

    __syncthreads();

    // Handle bias weights (ty == 0 && by == 0)
    if (ty == 0 && by == 0 && tx < hid) {
        float w_bias = w[index_x];
        float oldw_bias = oldw[index_x];
        float bias_update = (ETA * delta[index_x]) + (MOMENTUM * oldw_bias);
        
        w[index_x] = w_bias + bias_update;
        oldw[index_x] = bias_update;
    }
}
#endif
