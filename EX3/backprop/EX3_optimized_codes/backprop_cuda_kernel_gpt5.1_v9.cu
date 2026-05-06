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
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int index = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1);
    const int index_in = HEIGHT * by + ty + 1;

    __shared__ float input_node[HEIGHT];
    __shared__ float weight_matrix[HEIGHT][WIDTH];

    if (tx == 0) {
        // cache input once per row
        input_node[ty] = input_cuda[index_in];
    }

    __syncthreads();

    // coalesced load of weights
    weight_matrix[ty][tx] = input_hidden_cuda[index];

    __syncthreads();

    // fuse multiply into register, keep shared memory accesses minimal
    float val = weight_matrix[ty][tx] * input_node[ty];
    weight_matrix[ty][tx] = val;

    __syncthreads();

    // tree-reduction over ty dimension with power-of-two stride
    #pragma unroll
    for (int power_two = 2; power_two <= HEIGHT; power_two <<= 1) {
        if ((ty % power_two) == 0) {
            weight_matrix[ty][tx] += weight_matrix[ty + (power_two >> 1)][tx];
        }
        __syncthreads();
    }

    input_hidden_cuda[index] = weight_matrix[ty][tx];

    __syncthreads();

    if (tx == 0) {
        hidden_partial_sum[by * hid + ty] = weight_matrix[0][ty];
    }
}


__global__ void bpnn_adjust_weights_cuda(float *delta, int hid, float *ly,
                                         int in, float *w, float *oldw) {
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int index = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1);
    const int index_y = HEIGHT * by + ty + 1;
    const int index_x = tx + 1;

    const float d_x = delta[index_x];
    const float ly_y = ly[index_y];
    const float old_w_val = oldw[index];

    const float update = ETA * d_x * ly_y + MOMENTUM * old_w_val;

    w[index] += update;
    oldw[index] = update;

    __syncthreads();

    if (ty == 0 && by == 0) {
        const float old_w_x = oldw[index_x];
        const float update_x = ETA * d_x + MOMENTUM * old_w_x;
        w[index_x] += update_x;
        oldw[index_x] = update_x;
    }
}
#endif
