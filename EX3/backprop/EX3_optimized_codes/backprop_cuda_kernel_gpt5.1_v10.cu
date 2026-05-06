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

    int stride = hid + 1;

    int index = stride * HEIGHT * by + stride * ty + tx + 1 + stride;
    int index_in = HEIGHT * by + ty + 1;

    __shared__ float input_node[HEIGHT];
    __shared__ float weight_matrix[HEIGHT][WIDTH];

    if (tx == 0) {
        input_node[ty] = __ldg(&input_cuda[index_in]);
    }

    __syncthreads();

    weight_matrix[ty][tx] = __ldg(&input_hidden_cuda[index]);

    __syncthreads();

    weight_matrix[ty][tx] *= input_node[ty];

    __syncthreads();

    int lane = ty;
    for (int offset = HEIGHT >> 1; offset > 0; offset >>= 1) {
        if ((lane & ((offset << 1) - 1)) == 0 && (lane + offset) < HEIGHT) {
            weight_matrix[ty][tx] += weight_matrix[ty + offset][tx];
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

    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int stride = hid + 1;

    int index = stride * HEIGHT * by + stride * ty + tx + 1 + stride;
    int index_y = HEIGHT * by + ty + 1;
    int index_x = tx + 1;

    float delta_x = __ldg(&delta[index_x]);
    float ly_y = __ldg(&ly[index_y]);
    float old_w_val = oldw[index];
    float update = ETA * delta_x * ly_y + MOMENTUM * old_w_val;

    w[index] = w[index] + update;
    oldw[index] = update;

    __syncthreads();

    if (ty == 0 && by == 0) {
        float old_w_x = oldw[index_x];
        float update_x = ETA * delta_x + MOMENTUM * old_w_x;
        w[index_x] = w[index_x] + update_x;
        oldw[index_x] = update_x;
    }
}
#endif
