#ifndef _BACKPROP_CUDA_KERNEL_H_
#define _BACKPROP_CUDA_KERNEL_H_

#include <stdio.h>
#include "../backprop.h"
#include "math.h"
#include "cuda.h"

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

__global__ void bpnn_layerforward_CUDA(float *input_cuda,
                                       float *output_hidden_cuda,
                                       float *input_hidden_cuda,
                                       float *hidden_partial_sum, int in,
                                       int hid) {
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Precompute constants
    const int stride = hid + 1;
    const int base_matrix = stride * (HEIGHT * by + ty + 1) + (stride);
    const int index = base_matrix + tx + 1;
    const int index_in = HEIGHT * by + ty + 1;

    __shared__ float input_node[HEIGHT];
    __shared__ float weight_matrix[HEIGHT][WIDTH];

    // Coalesced load of input_node
    if (tx == 0) {
        input_node[ty] = input_cuda[index_in];
    }

    __syncthreads();

    // Coalesced load of weight_matrix
    weight_matrix[ty][tx] = input_hidden_cuda[index];

    __syncthreads();

    // Multiply by corresponding input node
    float val = weight_matrix[ty][tx] * input_node[ty];
    weight_matrix[ty][tx] = val;

    __syncthreads();

    // Tree reduction in shared memory along ty dimension
    // Assume HEIGHT is power of two as in original code
    for (int offset = HEIGHT >> 1; offset > 0; offset >>= 1) {
        if (ty < offset) {
            weight_matrix[ty][tx] += weight_matrix[ty + offset][tx];
        }
        __syncthreads();
    }

    // Write back per-connection partial sums
    input_hidden_cuda[index] = weight_matrix[ty][tx];

    __syncthreads();

    // Store partial sums for hidden layer (one per row)
    if (tx == 0) {
        hidden_partial_sum[by * hid + ty] = weight_matrix[0][ty];
    }
}


__global__ void bpnn_adjust_weights_cuda(float *delta, int hid, float *ly,
                                         int in, float *w, float *oldw) {

    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const int stride = hid + 1;
    const int base_matrix = stride * (HEIGHT * by + ty + 1) + (stride);
    const int index = base_matrix + tx + 1;
    const int index_y = HEIGHT * by + ty + 1;
    const int index_x = tx + 1;

    float delta_x = delta[index_x];
    float ly_y = ly[index_y];

    float prev_w = oldw[index];
    float update = ETA * delta_x * ly_y + MOMENTUM * prev_w;

    w[index] += update;
    oldw[index] = update;

    __syncthreads();

    // Bias update (single block-row/ty handled as in original)
    if (ty == 0 && by == 0) {
        float prev_w_x = oldw[index_x];
        float update_x = ETA * delta[index_x] + MOMENTUM * prev_w_x;
        w[index_x] += update_x;
        oldw[index_x] = update_x;
    }
}
#endif
