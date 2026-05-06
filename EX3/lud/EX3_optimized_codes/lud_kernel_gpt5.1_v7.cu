#include <cuda.h>
#include <stdio.h>
#include <time.h>

#include "../../common_rodinia/cuda/profile.h"

// Global variable for kernel timing
double g_kernel_time = 0.0;

#ifdef RD_WG_SIZE_0_0
#define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
#define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
#define BLOCK_SIZE RD_WG_SIZE
#else
#define BLOCK_SIZE 16
#endif


#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

// Using launch bounds to help compiler optimize for A100 occupancy
__global__ __launch_bounds__(BLOCK_SIZE)
void lud_diagonal(float * __restrict__ m, int matrix_dim, int offset) {
    int i, j;
    __shared__ float shadow[BLOCK_SIZE][BLOCK_SIZE];

    // Precompute threadIdx.x in a register
    const int tx = threadIdx.x;

    // Load diagonal block into shared memory with coalesced access
    int array_offset = offset * matrix_dim + offset;
#pragma unroll
    for (i = 0; i < BLOCK_SIZE; i++) {
        shadow[i][tx] = m[array_offset + tx];
        array_offset += matrix_dim;
    }
    __syncthreads();

#pragma unroll
    for (i = 0; i < BLOCK_SIZE - 1; i++) {

        if (tx > i) {
#pragma unroll
            for (j = 0; j < i; j++) {
                shadow[tx][i] -= shadow[tx][j] * shadow[j][i];
            }
            shadow[tx][i] /= shadow[i][i];
        }

        __syncthreads();

        if (tx > i) {
#pragma unroll
            for (j = 0; j < i + 1; j++) {
                shadow[i + 1][tx] -= shadow[i + 1][j] * shadow[j][tx];
            }
        }
        __syncthreads();
    }

    // Write back (first row unchanged, so start from i = 1)
    array_offset = (offset + 1) * matrix_dim + offset;
#pragma unroll
    for (i = 1; i < BLOCK_SIZE; i++) {
        m[array_offset + tx] = shadow[i][tx];
        array_offset += matrix_dim;
    }
}

__global__ __launch_bounds__(2 * BLOCK_SIZE)
void lud_perimeter(float * __restrict__ m, int matrix_dim, int offset) {
    __shared__ float dia[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float peri_row[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float peri_col[BLOCK_SIZE][BLOCK_SIZE];

    int i, j, array_offset;
    int idx;
    const int tx = threadIdx.x;

    if (tx < BLOCK_SIZE) {
        idx = tx;

        array_offset = offset * matrix_dim + offset;
#pragma unroll
        for (i = 0; i < BLOCK_SIZE / 2; i++) {
            dia[i][idx] = m[array_offset + idx];
            array_offset += matrix_dim;
        }

        array_offset = offset * matrix_dim + offset;
#pragma unroll
        for (i = 0; i < BLOCK_SIZE; i++) {
            peri_row[i][idx] =
                m[array_offset + (blockIdx.x + 1) * BLOCK_SIZE + idx];
            array_offset += matrix_dim;
        }

    } else {
        idx = tx - BLOCK_SIZE;

        array_offset = (offset + BLOCK_SIZE / 2) * matrix_dim + offset;
#pragma unroll
        for (i = BLOCK_SIZE / 2; i < BLOCK_SIZE; i++) {
            dia[i][idx] = m[array_offset + idx];
            array_offset += matrix_dim;
        }

        array_offset =
            (offset + (blockIdx.x + 1) * BLOCK_SIZE) * matrix_dim + offset;
#pragma unroll
        for (i = 0; i < BLOCK_SIZE; i++) {
            peri_col[i][idx] = m[array_offset + idx];
            array_offset += matrix_dim;
        }
    }
    __syncthreads();

    if (tx < BLOCK_SIZE) { // peri-row
        idx = tx;
#pragma unroll
        for (i = 1; i < BLOCK_SIZE; i++) {
#pragma unroll
            for (j = 0; j < i; j++) {
                peri_row[i][idx] -= dia[i][j] * peri_row[j][idx];
            }
        }
    } else { // peri-col
        idx = tx - BLOCK_SIZE;
#pragma unroll
        for (i = 0; i < BLOCK_SIZE; i++) {
#pragma unroll
            for (j = 0; j < i; j++) {
                peri_col[idx][i] -= peri_col[idx][j] * dia[j][i];
            }
            peri_col[idx][i] /= dia[i][i];
        }
    }

    __syncthreads();

    if (tx < BLOCK_SIZE) { // peri-row
        idx = tx;
        array_offset = (offset + 1) * matrix_dim + offset;
#pragma unroll
        for (i = 1; i < BLOCK_SIZE; i++) {
            m[array_offset + (blockIdx.x + 1) * BLOCK_SIZE + idx] =
                peri_row[i][idx];
            array_offset += matrix_dim;
        }
    } else { // peri-col
        idx = tx - BLOCK_SIZE;
        array_offset =
            (offset + (blockIdx.x + 1) * BLOCK_SIZE) * matrix_dim + offset;
#pragma unroll
        for (i = 0; i < BLOCK_SIZE; i++) {
            m[array_offset + idx] = peri_col[i][idx];
            array_offset += matrix_dim;
        }
    }
}

__global__ __launch_bounds__(BLOCK_SIZE * BLOCK_SIZE)
void lud_internal(float * __restrict__ m, int matrix_dim, int offset) {
    __shared__ float peri_row[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float peri_col[BLOCK_SIZE][BLOCK_SIZE];

    int i;
    float sum = 0.0f;

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    const int global_row_id = offset + (blockIdx.y + 1) * BLOCK_SIZE;
    const int global_col_id = offset + (blockIdx.x + 1) * BLOCK_SIZE;

    // Coalesced loads for perimeter row and column
    peri_row[ty][tx] =
        m[(offset + ty) * matrix_dim + global_col_id + tx];
    peri_col[ty][tx] =
        m[(global_row_id + ty) * matrix_dim + offset + tx];

    __syncthreads();

    // Compute inner product using registers
#pragma unroll
    for (i = 0; i < BLOCK_SIZE; i++) {
        sum += peri_col[ty][i] * peri_row[i][tx];
    }

    m[(global_row_id + ty) * matrix_dim + global_col_id + tx] -= sum;
}


void lud_cuda(float *m, int matrix_dim) {
    int i = 0;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    float *m_debug = (float *)malloc(matrix_dim * matrix_dim * sizeof(float));

    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    for (i = 0; i < matrix_dim - BLOCK_SIZE; i += BLOCK_SIZE) {
        PROFILE((
            lud_diagonal<<<1, BLOCK_SIZE>>>(m, matrix_dim, i)
        ));

        int grid_size = (matrix_dim - i) / BLOCK_SIZE - 1;

        PROFILE((
            lud_perimeter<<<grid_size, BLOCK_SIZE * 2>>>(
                m, matrix_dim, i)
        ));

        dim3 dimGrid(grid_size, grid_size);
        PROFILE((
            lud_internal<<<dimGrid, dimBlock>>>(m, matrix_dim, i)
        ));
    }
    PROFILE((
        lud_diagonal<<<1, BLOCK_SIZE>>>(m, matrix_dim, i)
    ));

    cudaDeviceSynchronize();
    
    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) + 
                         (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
    g_kernel_time += kernel_time;
}
