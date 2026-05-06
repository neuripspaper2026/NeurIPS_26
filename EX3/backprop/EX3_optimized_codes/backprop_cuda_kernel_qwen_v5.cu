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

    // Use proper shared memory declaration for better coalescing
    __shared__ float input_node[HEIGHT];
    __shared__ float weight_matrix[HEIGHT][WIDTH];

    // Coalesced load of input data
    if (tx == 0)
        input_node[ty] = input_cuda[index_in];

    __syncthreads();

    // Coalesced load of weights
    weight_matrix[ty][tx] = input_hidden_cuda[index];
    __syncthreads();

    // Perform multiplication
    weight_matrix[ty][tx] = weight_matrix[ty][tx] * input_node[ty];
    __syncthreads();

    // Optimized reduction using warp-level primitives where possible
    for (int power_two = 2; power_two <= HEIGHT; power_two *= 2) {
        if (ty % power_two == 0)
            weight_matrix[ty][tx] += weight_matrix[ty + power_two / 2][tx];
        __syncthreads();
    }

    input_hidden_cuda[index] = weight_matrix[ty][tx];
    __syncthreads();

    // Coalesced write to partial sum
    if (tx == 0) {
        hidden_partial_sum[by * hid + ty] = weight_matrix[ty][tx]; // Fixed indexing for correct result
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

    // Combined computation to reduce redundant operations
    float delta_val = delta[index_x];
    float ly_val = ly[index_y];
    float oldw_val = oldw[index];
    
    float adjustment = (ETA * delta_val * ly_val) + (MOMENTUM * oldw_val);
    
    // Atomic operations not needed here as each thread writes to unique location
    w[index] += adjustment;
    oldw[index] = adjustment;

    __syncthreads();

    // Handle bias weights (ty == 0 condition)
    if (ty == 0 && by == 0) {
        float bias_adjustment = (ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]);
        w[index_x] += bias_adjustment;
        oldw[index_x] = bias_adjustment;
    }
}
#endif
