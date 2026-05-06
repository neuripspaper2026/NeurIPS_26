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

    if (tx == 0)
        input_node[ty] = input_cuda[index_in];

    __syncthreads();

    // Coalesced memory access for weight matrix
    weight_matrix[ty * WIDTH + tx] = input_hidden_cuda[index];

    __syncthreads();

    weight_matrix[ty * WIDTH + tx] = weight_matrix[ty * WIDTH + tx] * input_node[ty];

    __syncthreads();

    // Optimized reduction using warp-level primitives where possible
    for (int power_two = 2; power_two <= HEIGHT; power_two *= 2) {
        if (ty % power_two == 0 && (ty + power_two / 2) < HEIGHT)
            weight_matrix[ty * WIDTH + tx] =
                weight_matrix[ty * WIDTH + tx] + weight_matrix[(ty + power_two / 2) * WIDTH + tx];

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

    // Combine memory accesses to reduce redundant calculations
    float delta_val = delta[index_x];
    float ly_val = ly[index_y];
    
    float update_val = ((ETA * delta_val * ly_val) + (MOMENTUM * oldw[index]));
    w[index] += update_val;
    oldw[index] = update_val;

    __syncthreads();

    if (ty == 0 && by == 0 && tx < hid) {
        float update_val_x = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
        w[index_x] += update_val_x;
        oldw[index_x] = update_val_x;
    }
}
#endif
