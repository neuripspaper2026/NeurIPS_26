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

    // Use linear shared memory for better memory access patterns
    extern __shared__ float shared_mem[];
    float* input_node = shared_mem;
    float* weight_matrix = &shared_mem[HEIGHT];

    // Coalesced load of input data
    if (tx == 0)
        input_node[ty] = input_cuda[index_in];

    __syncthreads();

    // Coalesced load of weights
    weight_matrix[ty * WIDTH + tx] = input_hidden_cuda[index];

    __syncthreads();

    // Perform computation
    weight_matrix[ty * WIDTH + tx] = weight_matrix[ty * WIDTH + tx] * input_node[ty];

    __syncthreads();

    // Optimized reduction using warp-level primitives where possible
    for (int power_two = 2; power_two <= HEIGHT; power_two *= 2) {
        if (ty % power_two == 0 && (ty + power_two/2) < HEIGHT)
            weight_matrix[ty * WIDTH + tx] += weight_matrix[(ty + power_two/2) * WIDTH + tx];
        __syncthreads();
    }

    // Write back results
    input_hidden_cuda[index] = weight_matrix[ty * WIDTH + tx];

    __syncthreads();

    // Write partial sums
    if (tx == 0) {
        hidden_partial_sum[by * hid + ty] = weight_matrix[tx * WIDTH + ty];
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

    // Use registers for frequently accessed values
    float delta_val = delta[index_x];
    float ly_val = ly[index_y];
    float oldw_val = oldw[index];
    float w_val = w[index];

    // Combined computation to reduce memory traffic
    float adjustment = (ETA * delta_val * ly_val) + (MOMENTUM * oldw_val);
    w_val += adjustment;
    oldw_val = adjustment;

    // Write back results
    w[index] = w_val;
    oldw[index] = oldw_val;

    __syncthreads();

    // Handle bias weights
    if (ty == 0 && by == 0) {
        float oldw_x_val = oldw[index_x];
        float adjustment_x = (ETA * delta_val) + (MOMENTUM * oldw_x_val);
        w[index_x] += adjustment_x;
        oldw[index_x] = adjustment_x;
    }
}
#endif
