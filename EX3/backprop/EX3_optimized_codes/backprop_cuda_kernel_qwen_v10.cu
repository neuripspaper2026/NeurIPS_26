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

    // Use shared memory with proper padding for bank conflict avoidance
    __shared__ float input_node[HEIGHT];
    __shared__ float weight_matrix[HEIGHT][WIDTH+1]; // Padding to avoid bank conflicts

    // Coalesced memory access for input_node
    if (tx == 0)
        input_node[ty] = input_cuda[index_in];

    __syncthreads();

    // Coalesced memory access for weight_matrix
    weight_matrix[ty][tx] = input_hidden_cuda[index];
    __syncthreads();

    // Perform multiplication
    weight_matrix[ty][tx] = weight_matrix[ty][tx] * input_node[ty];
    __syncthreads();

    // Optimized reduction with warp-level primitives where possible
    for (int power_two = 2; power_two <= HEIGHT; power_two *= 2) {
        if (ty % power_two == 0 && (ty + power_two/2) < HEIGHT)
            weight_matrix[ty][tx] += weight_matrix[ty + power_two/2][tx];
        __syncthreads();
    }

    // Write back results
    input_hidden_cuda[index] = weight_matrix[ty][tx];

    // Ensure bounds checking for partial sum write
    if (tx == 0 && ty < hid) {
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

    // Use registers to reduce global memory access
    float w_val = w[index];
    float oldw_val = oldw[index];
    float ly_val = ly[index_y];
    float delta_val = delta[index_x];
    
    float adjustment = ETA * delta_val * ly_val;
    float momentum_term = MOMENTUM * oldw_val;
    float weight_update = adjustment + momentum_term;
    
    w[index] = w_val + weight_update;
    oldw[index] = weight_update;

    __syncthreads();

    // Handle bias weights with bounds checking
    if (ty == 0 && by == 0 && index_x < (hid + 1)) {
        float w_bias = w[index_x];
        float oldw_bias = oldw[index_x];
        float bias_update = (ETA * delta_val) + (MOMENTUM * oldw_bias);
        
        w[index_x] = w_bias + bias_update;
        oldw[index_x] = bias_update;
    }
}
#endif
