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
    int base   = stride * (HEIGHT * by + ty + 1) + 1 + stride;

    int index    = base + tx;
    int index_in = HEIGHT * by + ty + 1;

    __shared__ float input_node[HEIGHT];
    __shared__ float weight_matrix[HEIGHT][WIDTH];

    if (tx == 0) {
        input_node[ty] = __ldg(&input_cuda[index_in]);
    }

    __syncthreads();

    weight_matrix[ty][tx] = __ldg(&input_hidden_cuda[index]);

    __syncthreads();

    float val = weight_matrix[ty][tx] * input_node[ty];
    weight_matrix[ty][tx] = val;

    __syncthreads();

    // warp-level reduction across ty dimension when HEIGHT == 32
#if (HEIGHT == 32)
    unsigned mask = 0xffffffff;
    float sum = val;
    // reduce over ty (threads in y dimension share same tx)
    // reinterpret ty as lane id in a warp
    for (int offset = HEIGHT / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(mask, sum, offset, HEIGHT);
        if (ty + offset < HEIGHT)
            sum += other;
    }
    if (ty == 0) {
        weight_matrix[0][tx] = sum;
    }
    __syncthreads();
#else
    for (int power_two = 2; power_two <= HEIGHT; power_two <<= 1) {
        if ((ty % power_two) == 0) {
            weight_matrix[ty][tx] += weight_matrix[ty + power_two / 2][tx];
        }
        __syncthreads();
    }
#endif

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
    int base   = stride * (HEIGHT * by + ty + 1) + 1 + stride;

    int index   = base + tx;
    int index_y = HEIGHT * by + ty + 1;
    int index_x = tx + 1;

    float d  = __ldg(&delta[index_x]);
    float lyv = __ldg(&ly[index_y]);
    float old = __ldg(&oldw[index]);

    float grad = ETA * d * lyv + MOMENTUM * old;

    w[index]    = w[index] + grad;
    oldw[index] = grad;

    __syncthreads();

    if (ty == 0 && by == 0) {
        float old_root = __ldg(&oldw[index_x]);
        float grad_root = ETA * d + MOMENTUM * old_root;
        w[index_x]    = w[index_x] + grad_root;
        oldw[index_x] = grad_root;
    }
}
#endif
