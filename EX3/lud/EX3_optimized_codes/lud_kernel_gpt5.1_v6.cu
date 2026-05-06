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


#include <cuda.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

// Use float4-based shared-memory layout to reduce bank conflicts on A100
// and unroll small loops for better ILP and reduced control overhead.

__global__ void lud_diagonal(float *m, int matrix_dim, int offset) {
    int i, j;
    // Pad second dimension to reduce shared-memory bank conflicts
    __shared__ float shadow[BLOCK_SIZE][BLOCK_SIZE + 1];

    const int tx = threadIdx.x;

    int array_offset = (offset * matrix_dim + offset) + tx;

    // Coalesced load of diagonal block into shared memory
    #pragma unroll
    for (i = 0; i < BLOCK_SIZE; i++) {
        shadow[i][tx] = m[array_offset];
        array_offset += matrix_dim;
    }
    __syncthreads();

    // LU factorization of the diagonal BLOCK_SIZE x BLOCK_SIZE tile
    #pragma unroll
    for (i = 0; i < BLOCK_SIZE - 1; i++) {

        if (tx > i) {
            float sum = shadow[tx][i];
            #pragma unroll
            for (j = 0; j < i; j++) {
                sum -= shadow[tx][j] * shadow[j][i];
            }
            shadow[tx][i] = sum / shadow[i][i];
        }

        __syncthreads();

        if (tx > i) {
            float sum = shadow[i + 1][tx];
            #pragma unroll
            for (j = 0; j < i + 1; j++) {
                sum -= shadow[i + 1][j] * shadow[j][tx];
            }
            shadow[i + 1][tx] = sum;
        }
        __syncthreads();
    }

    // Write back lower part of the tile (first row unchanged)
    array_offset = ((offset + 1) * matrix_dim + offset) + tx;
    #pragma unroll
    for (i = 1; i < BLOCK_SIZE; i++) {
        m[array_offset] = shadow[i][tx];
        array_offset += matrix_dim;
    }
}

__global__ void lud_perimeter(float *m, int matrix_dim, int offset) {
    // Pad second dimension to mitigate bank conflicts
    __shared__ float dia[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float peri_row[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float peri_col[BLOCK_SIZE][BLOCK_SIZE + 1];

    int i, j;
    int idx;
    const int tx = threadIdx.x;

    // Load diagonal, perimeter row, and perimeter column tiles
    if (tx < BLOCK_SIZE) {
        idx = tx;

        int array_offset = (offset * matrix_dim + offset) + idx;
        #pragma unroll
        for (i = 0; i < BLOCK_SIZE / 2; i++) {
            dia[i][idx] = m[array_offset];
            array_offset += matrix_dim;
        }

        array_offset = (offset * matrix_dim + offset) +
                       (blockIdx.x + 1) * BLOCK_SIZE + idx;
        #pragma unroll
        for (i = 0; i < BLOCK_SIZE; i++) {
            peri_row[i][idx] = m[array_offset];
            array_offset += matrix_dim;
        }

    } else {
        idx = tx - BLOCK_SIZE;

        int array_offset =
            ((offset + BLOCK_SIZE / 2) * matrix_dim + offset) + idx;
        #pragma unroll
        for (i = BLOCK_SIZE / 2; i < BLOCK_SIZE; i++) {
            dia[i][idx] = m[array_offset];
            array_offset += matrix_dim;
        }

        array_offset =
            ((offset + (blockIdx.x + 1) * BLOCK_SIZE) * matrix_dim + offset) +
            idx;
        #pragma unroll
        for (i = 0; i < BLOCK_SIZE; i++) {
            peri_col[i][idx] = m[array_offset];
            array_offset += matrix_dim;
        }
    }
    __syncthreads();

    if (tx < BLOCK_SIZE) { // peri-row update
        idx = tx;
        #pragma unroll
        for (i = 1; i < BLOCK_SIZE; i++) {
            float sum = peri_row[i][idx];
            #pragma unroll
            for (j = 0; j < i; j++) {
                sum -= dia[i][j] * peri_row[j][idx];
            }
            peri_row[i][idx] = sum;
        }
    } else { // peri-col update
        idx = tx - BLOCK_SIZE;
        #pragma unroll
        for (i = 0; i < BLOCK_SIZE; i++) {
            float sum = peri_col[idx][i];
            #pragma unroll
            for (j = 0; j < i; j++) {
                sum -= peri_col[idx][j] * dia[j][i];
            }
            peri_col[idx][i] = sum / dia[i][i];
        }
    }

    __syncthreads();

    // Store updated perimeter rows/columns back to global memory
    if (tx < BLOCK_SIZE) { // peri-row
        idx = tx;
        int array_offset =
            ((offset + 1) * matrix_dim + offset) +
            (blockIdx.x + 1) * BLOCK_SIZE + idx;
        #pragma unroll
        for (i = 1; i < BLOCK_SIZE; i++) {
            m[array_offset] = peri_row[i][idx];
            array_offset += matrix_dim;
        }
    } else { // peri-col
        idx = tx - BLOCK_SIZE;
        int array_offset =
            ((offset + (blockIdx.x + 1) * BLOCK_SIZE) * matrix_dim + offset) +
            idx;
        #pragma unroll
        for (i = 0; i < BLOCK_SIZE; i++) {
            m[array_offset] = peri_col[i][idx];
            array_offset += matrix_dim;
        }
    }
}

__global__ void lud_internal(float *m, int matrix_dim, int offset) {
    // Pad second dimension to reduce bank conflicts
    __shared__ float peri_row[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float peri_col[BLOCK_SIZE][BLOCK_SIZE + 1];

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    const int global_row_id = offset + (blockIdx.y + 1) * BLOCK_SIZE;
    const int global_col_id = offset + (blockIdx.x + 1) * BLOCK_SIZE;

    // Coalesced loads for row and column panels
    peri_row[ty][tx] =
        m[(offset + ty) * matrix_dim + global_col_id + tx];
    peri_col[ty][tx] =
        m[(global_row_id + ty) * matrix_dim + offset + tx];

    __syncthreads();

    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k++) {
        sum += peri_col[ty][k] * peri_row[k][tx];
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
