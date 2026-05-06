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
    // Assumptions (from original code):
    // blockDim.x == WIDTH, blockDim.y == HEIGHT
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Precompute commonly used values
    int stride_hid = hid + 1;
    int base_mat   = stride_hid * (HEIGHT * by + ty + 1 + HEIGHT);
    int index      = base_mat + tx;
    int index_in   = HEIGHT * by + ty + 1;

    __shared__ float input_node[HEIGHT];
    __shared__ float weight_matrix[HEIGHT][WIDTH];

    // Load input node once per row (broadcast within the row)
    if (tx == 0) {
        input_node[ty] = __ldg(&input_cuda[index_in]);
    }

    __syncthreads();

    // Coalesced load of weight matrix
    weight_matrix[ty][tx] = __ldg(&input_hidden_cuda[index]);

    __syncthreads();

    // Elementwise multiply
    float val = weight_matrix[ty][tx] * input_node[ty];
    weight_matrix[ty][tx] = val;

    __syncthreads();

    // Parallel reduction along ty dimension for each tx
    // Unroll the small fixed reduction over HEIGHT if known at compile time.
#pragma unroll
    for (int power_two = 2; power_two <= HEIGHT; power_two *= 2) {
        if ((ty & (power_two - 1)) == 0) {
            weight_matrix[ty][tx] += weight_matrix[ty + (power_two >> 1)][tx];
        }
        __syncthreads();
    }

    // Write back partial sums
    input_hidden_cuda[index] = weight_matrix[ty][tx];

    __syncthreads();

    // Store final reduction result per row into hidden_partial_sum
    if (tx == 0) {
        hidden_partial_sum[by * hid + ty] = weight_matrix[0][ty];
    }
}


__global__ void bpnn_adjust_weights_cuda(float *delta, int hid, float *ly,
                                         int in, float *w, float *oldw) {

    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int stride_hid = hid + 1;
    int base_mat   = stride_hid * (HEIGHT * by + ty + 1 + HEIGHT);
    int index      = base_mat + tx;
    int index_y    = HEIGHT * by + ty + 1;
    int index_x    = tx + 1;

    float d  = __ldg(&delta[index_x]);
    float ly_val = __ldg(&ly[index_y]);
    float old    = oldw[index];
    float grad   = ETA * d * ly_val + MOMENTUM * old;

    w[index]    += grad;
    oldw[index]  = grad;

    __syncthreads();

    // Update bias weights (only once)
    if (ty == 0 && by == 0) {
        float db   = __ldg(&delta[index_x]);
        float oldb = oldw[index_x];
        float gradb = ETA * db + MOMENTUM * oldb;
        w[index_x]    += gradb;
        oldw[index_x]  = gradb;
    }
}
#endif
