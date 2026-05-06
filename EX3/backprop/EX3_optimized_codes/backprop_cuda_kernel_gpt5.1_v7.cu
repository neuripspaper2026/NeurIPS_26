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
    int base = stride * (HEIGHT * by + ty + 1 + 1);

    int index = base + tx;

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

    // Warp-level reduction along ty dimension when HEIGHT <= 32
#if (HEIGHT <= 32)
    // reinterpret shared row as float pointer for this column
    float val = weight_matrix[ty][tx];
    // Use full warp mask (HEIGHT is assumed power of two in original code)
    unsigned mask = 0xffffffffu;

    // Perform warp shuffle reduction along ty (threads in same warp, same tx)
    for (int offset = HEIGHT >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }

    if (ty == 0) {
        weight_matrix[0][tx] = val;
    }
    __syncthreads();
#else
    for (int power_two = 2; power_two <= HEIGHT; power_two *= 2) {

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
    int base = stride * (HEIGHT * by + ty + 1 + 1);
    int index = base + tx;

    int index_y = HEIGHT * by + ty + 1;
    int index_x = tx + 1;
    // eta = 0.3;
    // momentum = 0.3;

    float d = __ldg(&delta[index_x]);
    float l = __ldg(&ly[index_y]);
    float old = oldw[index];
    float update = ETA * d * l + MOMENTUM * old;

    w[index] += update;
    oldw[index] = update;

    __syncthreads();

    if (ty == 0 && by == 0) {
        float d_root = __ldg(&delta[index_x]);
        float old_root = oldw[index_x];
        float upd_root = ETA * d_root + MOMENTUM * old_root;
        w[index_x] += upd_root;
        oldw[index_x] = upd_root;
    }
}
#endif
