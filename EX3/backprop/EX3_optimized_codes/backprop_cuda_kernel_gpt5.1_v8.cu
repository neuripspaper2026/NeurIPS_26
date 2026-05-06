#ifndef _BACKPROP_CUDA_KERNEL_H_
#define _BACKPROP_CUDA_KERNEL_H_

#include <stdio.h>
#include "../backprop.h"
#include "math.h"
#include "cuda.h"


__global__ void bpnn_layerforward_CUDA(float * __restrict__ input_cuda,
                                       float * __restrict__ output_hidden_cuda,
                                       float * __restrict__ input_hidden_cuda,
                                       float * __restrict__ hidden_partial_sum,
                                       int in,
                                       int hid) {
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int stride = hid + 1;

    // Precompute commonly used indices
    const int base_weight = stride * HEIGHT * by + stride * ty + tx + 1 + stride;
    const int index_in    = HEIGHT * by + ty + 1;

    __shared__ float input_node[HEIGHT];
    __shared__ float weight_matrix[HEIGHT][WIDTH];

    // Load input once per row (coalesced across ty when tx==0)
    if (tx == 0) {
        input_node[ty] = __ldg(&input_cuda[index_in]);
    }

    __syncthreads();

    // Coalesced load of weight matrix from global to shared
    weight_matrix[ty][tx] = __ldg(&input_hidden_cuda[base_weight]);

    __syncthreads();

    // Multiply by corresponding input node (all threads in row use same value)
    float val = weight_matrix[ty][tx] * input_node[ty];
    weight_matrix[ty][tx] = val;

    __syncthreads();

    // Tree-based reduction along ty dimension using shared memory
    // Assume HEIGHT is power-of-two as in the original code
    for (int power_two = 2; power_two <= HEIGHT; power_two <<= 1) {
        if ((ty & (power_two - 1)) == 0) {
            weight_matrix[ty][tx] += weight_matrix[ty + (power_two >> 1)][tx];
        }
        __syncthreads();
    }

    // Write back partial results to global memory
    input_hidden_cuda[base_weight] = weight_matrix[ty][tx];

    __syncthreads();

    // Final partial sum stored by first column threads
    if (tx == 0) {
        hidden_partial_sum[by * hid + ty] = weight_matrix[0][ty];
    }
}


__global__ void bpnn_adjust_weights_cuda(float * __restrict__ delta,
                                         int hid,
                                         float * __restrict__ ly,
                                         int in,
                                         float * __restrict__ w,
                                         float * __restrict__ oldw) {
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int stride = hid + 1;

    const int index    = stride * HEIGHT * by + stride * ty + tx + 1 + stride;
    const int index_y  = HEIGHT * by + ty + 1;
    const int index_x  = tx + 1;

    // Read inputs once and reuse (avoid multiple global reads)
    const float d  = __ldg(&delta[index_x]);
    const float lyv = __ldg(&ly[index_y]);
    const float old = oldw[index];

    const float grad = ETA * d * lyv + MOMENTUM * old;

    w[index]    = w[index] + grad;
    oldw[index] = grad;

    __syncthreads();

    // Bias weight update (shared across blocks in original; keep semantics)
    if (ty == 0 && by == 0) {
        const float old_bias = oldw[index_x];
        const float grad_bias = ETA * d + MOMENTUM * old_bias;
        w[index_x]    = w[index_x] + grad_bias;
        oldw[index_x] = grad_bias;
    }
}
#endif
