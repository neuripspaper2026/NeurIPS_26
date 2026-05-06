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

    // Coalesced load: all threads in a warp load contiguous memory
    if (tx == 0)
        input_node[ty] = input_cuda[index_in];

    __syncthreads();

    // Coalesced load from global memory
    weight_matrix[ty][tx] = input_hidden_cuda[index];

    __syncthreads();

    // Element-wise multiplication
    weight_matrix[ty][tx] = weight_matrix[ty][tx] * input_node[ty];

    __syncthreads();

    // Optimized reduction using warp shuffle for the first 32 elements
    // then tree reduction for remaining
    float val = weight_matrix[ty][tx];
    
    // Warp-level reduction for ty dimension when HEIGHT <= 32
    #if HEIGHT <= 32
    for (int offset = HEIGHT / 2; offset > 0; offset >>= 1) {
        if (ty < offset) {
            val += weight_matrix[ty + offset][tx];
            weight_matrix[ty][tx] = val;
        }
        __syncthreads();
    }
    #else
    // Tree reduction for larger HEIGHT
    for (int power_two = 2; power_two <= HEIGHT; power_two <<= 1) {
        if (ty % power_two == 0) {
            val += weight_matrix[ty + power_two / 2][tx];
            weight_matrix[ty][tx] = val;
        }
        __syncthreads();
    }
    #endif

    // Coalesced write back to global memory
    input_hidden_cuda[index] = weight_matrix[ty][tx];

    __syncthreads();

    // Coalesced write to hidden_partial_sum
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

    // Preload delta and ly values to registers for reuse
    float delta_val = delta[index_x];
    float ly_val = ly[index_y];
    float oldw_val = oldw[index];
    
    // Compute new weight update once
    float weight_update = (ETA * delta_val * ly_val) + (MOMENTUM * oldw_val);
    
    // Coalesced writes to global memory
    w[index] += weight_update;
    oldw[index] = weight_update;

    __syncthreads();

    // Bias update - only one thread per block updates
    if (ty == 0 && by == 0) {
        float bias_update = (ETA * delta_val) + (MOMENTUM * oldw[index_x]);
        w[index_x] += bias_update;
        oldw[index_x] = bias_update;
    }
}
#endif
