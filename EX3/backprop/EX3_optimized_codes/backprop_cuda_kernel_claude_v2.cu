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

    __shared__ float input_node[HEIGHT];
    __shared__ float weight_matrix[HEIGHT][WIDTH];

    // Coalesced load: all threads in warp load contiguous memory
    if (tx == 0)
        input_node[ty] = input_cuda[index_in];

    __syncthreads();

    // Coalesced load from global memory
    weight_matrix[ty][tx] = input_hidden_cuda[index];

    __syncthreads();

    // Element-wise multiplication
    weight_matrix[ty][tx] = weight_matrix[ty][tx] * input_node[ty];

    __syncthreads();

    // Optimized reduction using warp-level primitives for A100
    // First reduce within warps using shuffle operations
    float val = weight_matrix[ty][tx];
    
    // Warp-level reduction (assuming WIDTH <= 32 for warp shuffle)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    
    // Store back to shared memory only from lane 0 of each warp
    if (tx == 0) {
        weight_matrix[ty][0] = val;
    }
    
    __syncthreads();

    // Traditional tree reduction for remaining elements if WIDTH > 32
    if (WIDTH > 32) {
        for (int power_two = 64; power_two <= HEIGHT; power_two *= 2) {
            if (ty % power_two == 0 && tx == 0)
                weight_matrix[ty][0] = weight_matrix[ty][0] + weight_matrix[ty + power_two / 2][0];
            __syncthreads();
        }
    } else {
        // If WIDTH <= 32, complete reduction across HEIGHT dimension
        for (int power_two = 2; power_two <= HEIGHT; power_two *= 2) {
            if (ty % power_two == 0 && tx == 0)
                weight_matrix[ty][0] = weight_matrix[ty][0] + weight_matrix[ty + power_two / 2][0];
            __syncthreads();
        }
    }

    // Write results with coalesced access pattern
    if (tx == 0) {
        input_hidden_cuda[index] = weight_matrix[ty][0];
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

    // Preload frequently accessed values into registers
    float delta_val = delta[index_x];
    float ly_val = ly[index_y];
    float oldw_val = oldw[index];
    
    // Compute update value once
    float update = (ETA * delta_val * ly_val) + (MOMENTUM * oldw_val);
    
    // Coalesced writes to global memory
    w[index] += update;
    oldw[index] = update;

    __syncthreads();

    // Reduce divergence by using predication
    if (ty == 0 && by == 0) {
        float bias_update = (ETA * delta_val) + (MOMENTUM * oldw[index_x]);
        w[index_x] += bias_update;
        oldw[index_x] = bias_update;
    }
}
#endif
