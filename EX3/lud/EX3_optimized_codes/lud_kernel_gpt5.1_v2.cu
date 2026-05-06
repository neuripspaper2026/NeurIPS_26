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


#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

// Use launch bounds to help compiler optimize for A100 SMs
__global__ __launch_bounds__(BLOCK_SIZE, 8)
void lud_diagonal(float * __restrict__ m, int matrix_dim, int offset) {
    int i, j;
    __shared__ float shadow[BLOCK_SIZE][BLOCK_SIZE + 1]; // pad to avoid bank conflicts

    const int tx = threadIdx.x;
    int array_offset = offset * matrix_dim + offset;

    // Coalesced loads into shared memory; each thread loads one column element per row
    #pragma unroll
    for (i = 0; i < BLOCK_SIZE; i++) {
        shadow[i][tx] = m[array_offset + tx];
        array_offset += matrix_dim;
    }
    __syncthreads();

    // LU decomposition of diagonal block
    #pragma unroll
    for (i = 0; i < BLOCK_SIZE - 1; i++) {

        if (tx > i) {
            float sum = shadow[tx][i];
            // Use FMA in the inner loop; unroll for small BLOCK_SIZE (typical: 16)
            #pragma unroll
            for (j = 0; j < BLOCK_SIZE; j++) {
                if (j >= i) break;
                sum = fmaf(-shadow[tx][j], shadow[j][i], sum);
            }
            shadow[tx][i] = sum * __frcp_rn(shadow[i][i]);
        }

        __syncthreads();

        if (tx > i) {
            float sum2 = shadow[i + 1][tx];
            #pragma unroll
            for (j = 0; j < BLOCK_SIZE; j++) {
                if (j > i) break;
                sum2 = fmaf(-shadow[i + 1][j], shadow[j][tx], sum2);
            }
            shadow[i + 1][tx] = sum2;
        }
        __syncthreads();
    }

    // Write back (excluding first row which is unchanged)
    array_offset = (offset + 1) * matrix_dim + offset;
    #pragma unroll
    for (i = 1; i < BLOCK_SIZE; i++) {
        m[array_offset + tx] = shadow[i][tx];
        array_offset += matrix_dim;
    }
}

__global__ __launch_bounds__(BLOCK_SIZE * 2, 4)
void lud_perimeter(float * __restrict__ m, int matrix_dim, int offset) {
    __shared__ float dia[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float peri_row[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float peri_col[BLOCK_SIZE][BLOCK_SIZE + 1];

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
        // Forward substitution using dia (lower) on peri_row
        #pragma unroll
        for (i = 1; i < BLOCK_SIZE; i++) {
            float sum = peri_row[i][idx];
            #pragma unroll
            for (j = 0; j < BLOCK_SIZE; j++) {
                if (j >= i) break;
                sum = fmaf(-dia[i][j], peri_row[j][idx], sum);
            }
            peri_row[i][idx] = sum;
        }
    } else { // peri-col
        idx = tx - BLOCK_SIZE;
        // Backward substitution using dia (upper) on peri_col
        #pragma unroll
        for (i = 0; i < BLOCK_SIZE; i++) {
            float sum = peri_col[idx][i];
            #pragma unroll
            for (j = 0; j < BLOCK_SIZE; j++) {
                if (j >= i) break;
                sum = fmaf(-peri_col[idx][j], dia[j][i], sum);
            }
            peri_col[idx][i] = sum * __frcp_rn(dia[i][i]);
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

__global__ __launch_bounds__(BLOCK_SIZE * BLOCK_SIZE, 2)
void lud_internal(float * __restrict__ m, int matrix_dim, int offset) {
    __shared__ float peri_row[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float peri_col[BLOCK_SIZE][BLOCK_SIZE + 1];

    int i;
    float sum;

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    const int global_row_id = offset + (blockIdx.y + 1) * BLOCK_SIZE;
    const int global_col_id = offset + (blockIdx.x + 1) * BLOCK_SIZE;

    // Coalesced loads of perimeter blocks
    peri_row[ty][tx] =
        m[(offset + ty) * matrix_dim + global_col_id + tx];
    peri_col[ty][tx] =
        m[(global_row_id + ty) * matrix_dim + offset + tx];

    __syncthreads();

    // Compute update using dot product; unroll and use FMAs
    sum = 0.0f;
    #pragma unroll
    for (i = 0; i < BLOCK_SIZE; i++) {
        sum = fmaf(peri_col[ty][i], peri_row[i][tx], sum);
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
