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

    // Coalesced memory access for input_node
    if (tx == 0)
        input_node[ty] = input_cuda[index_in];

    __syncthreads();

    // Linearized access to shared memory
    weight_matrix[ty * WIDTH + tx] = input_hidden_cuda[index];

    __syncthreads();

    // Vectorized computation
    weight_matrix[ty * WIDTH + tx] = weight_matrix[ty * WIDTH + tx] * input_node[ty];

    __syncthreads();

    // Optimized reduction with warp-level primitives where possible
    for (int power_two = 2; power_two <= HEIGHT; power_two *= 2) {
        if (ty % power_two == 0 && (ty + power_two/2) < HEIGHT)
            weight_matrix[ty * WIDTH + tx] += weight_matrix[(ty + power_two/2) * WIDTH + tx];
        __syncthreads();
    }

    input_hidden_cuda[index] = weight_matrix[ty * WIDTH + tx];

    __syncthreads();

    if (tx == 0 && ty < hid) {
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

    // Use registers to reduce global memory accesses
    float w_val = w[index];
    float oldw_val = oldw[index];
    float ly_val = ly[index_y];
    float delta_val = delta[index_x];
    
    float adjustment = (ETA * delta_val * ly_val) + (MOMENTUM * oldw_val);
    
    w[index] = w_val + adjustment;
    oldw[index] = adjustment;

    __syncthreads();

    if (ty == 0 && by == 0 && tx < hid) {
        float w0_val = w[index_x];
        float oldw0_val = oldw[index_x];
        float delta0_val = delta[index_x];
        
        float adjustment0 = (ETA * delta0_val) + (MOMENTUM * oldw0_val);
        
        w[index_x] = w0_val + adjustment0;
        oldw[index_x] = adjustment0;
    }
}
#endif
