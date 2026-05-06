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

    // Use linearized shared memory for better memory access patterns
    __shared__ float input_node[HEIGHT];
    __shared__ float weight_matrix[HEIGHT * WIDTH];

    // Coalesced load of input_node when tx == 0
    if (tx == 0)
        input_node[ty] = input_cuda[index_in];

    __syncthreads();

    // Coalesced load of weight_matrix
    weight_matrix[ty * WIDTH + tx] = input_hidden_cuda[index];

    __syncthreads();

    // Perform multiplication
    weight_matrix[ty * WIDTH + tx] = weight_matrix[ty * WIDTH + tx] * input_node[ty];

    __syncthreads();

    // Optimized reduction using sequential addressing
    for (int stride = 1; stride < HEIGHT; stride *= 2) {
        if (ty % (2 * stride) == 0 && (ty + stride) < HEIGHT) {
            weight_matrix[ty * WIDTH + tx] += weight_matrix[(ty + stride) * WIDTH + tx];
        }
        __syncthreads();
    }

    // Write back result
    input_hidden_cuda[index] = weight_matrix[ty * WIDTH + tx];

    __syncthreads();

    // Coalesced write to hidden_partial_sum
    if (tx == 0) {
        hidden_partial_sum[by * hid + ty] = weight_matrix[0 * WIDTH + ty];
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

    // Perform computation
    float delta_val = ETA * delta[index_x] * ly[index_y];
    float momentum_val = MOMENTUM * oldw[index];
    float adjustment = delta_val + momentum_val;

    // Update weights
    w[index] += adjustment;
    oldw[index] = adjustment;

    __syncthreads();

    // Handle bias weights
    if (ty == 0 && by == 0) {
        float bias_delta = ETA * delta[index_x];
        float bias_momentum = MOMENTUM * oldw[index_x];
        float bias_adjustment = bias_delta + bias_momentum;
        
        w[index_x] += bias_adjustment;
        oldw[index_x] = bias_adjustment;
    }
}
#endif
