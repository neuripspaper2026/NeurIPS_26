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

    // Use linear shared memory for better coalescing and A100 L1 cache utilization
    __shared__ float input_node[HEIGHT];
    __shared__ float weight_matrix[HEIGHT * WIDTH];

    // Coalesced load of input node values
    if (tx == 0)
        input_node[ty] = input_cuda[index_in];

    __syncthreads();

    // Linear indexing for shared memory
    weight_matrix[ty * WIDTH + tx] = input_hidden_cuda[index];

    __syncthreads();

    // Vectorized computation
    weight_matrix[ty * WIDTH + tx] = weight_matrix[ty * WIDTH + tx] * input_node[ty];

    __syncthreads();

    // Optimized reduction with better warp utilization
    for (int power_two = 2; power_two <= HEIGHT; power_two *= 2) {
        int mate = ty + power_two / 2;
        if (mate < HEIGHT && ty % power_two == 0) {
            weight_matrix[ty * WIDTH + tx] += weight_matrix[mate * WIDTH + tx];
        }
        __syncthreads();
    }

    input_hidden_cuda[index] = weight_matrix[ty * WIDTH + tx];

    __syncthreads();

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
    float weight_update = (ETA * delta_val * ly_val) + (MOMENTUM * oldw_val);
    
    w[index] = w_val + weight_update;
    oldw[index] = weight_update;

    __syncthreads();

    if (ty == 0 && by == 0) {
        float oldw_x = oldw[index_x];
        float w_x = w[index_x];
        float delta_x = delta[index_x];
        
        float update = (ETA * delta_x) + (MOMENTUM * oldw_x);
        w[index_x] = w_x + update;
        oldw[index_x] = update;
    }
}
#endif
