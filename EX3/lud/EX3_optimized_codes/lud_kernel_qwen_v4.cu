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


__global__ void lud_diagonal(float *m, int matrix_dim, int offset) {
    int i, j;
    __shared__ float shadow[BLOCK_SIZE][BLOCK_SIZE];

    int array_offset = offset * matrix_dim + offset;
    // Coalesced loading using all threads in block
    for (i = threadIdx.x; i < BLOCK_SIZE * BLOCK_SIZE; i += blockDim.x) {
        int row = i / BLOCK_SIZE;
        int col = i % BLOCK_SIZE;
        shadow[row][col] = m[array_offset + row * matrix_dim + col];
    }
    __syncthreads();

    for (i = 0; i < BLOCK_SIZE - 1; i++) {
        if (threadIdx.x > i) {
            float sum = 0.0f;
            for (j = 0; j < i; j++)
                sum += shadow[threadIdx.x][j] * shadow[j][i];
            shadow[threadIdx.x][i] -= sum;
            shadow[threadIdx.x][i] /= shadow[i][i];
        }
        __syncthreads();

        if (threadIdx.x > i) {
            float sum = 0.0f;
            for (j = 0; j < i + 1; j++)
                sum += shadow[i + 1][j] * shadow[j][threadIdx.x];
            shadow[i + 1][threadIdx.x] -= sum;
        }
        __syncthreads();
    }

    // Coalesced writing using all threads in block
    array_offset = (offset + 1) * matrix_dim + offset;
    for (i = 1 + threadIdx.x; i < BLOCK_SIZE; i += blockDim.x) {
        m[array_offset + (i - 1) * matrix_dim + threadIdx.x] = shadow[i][threadIdx.x];
    }
}

__global__ void lud_perimeter(float *m, int matrix_dim, int offset) {
    __shared__ float dia[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float peri_row[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float peri_col[BLOCK_SIZE][BLOCK_SIZE];

    int i, j, array_offset;
    int idx;

    // Coalesced loading of diagonal elements
    if (threadIdx.x < BLOCK_SIZE) {
        idx = threadIdx.x;
        // Load first half of diagonal
        for (i = threadIdx.x; i < BLOCK_SIZE * BLOCK_SIZE / 2; i += blockDim.x) {
            int row = i / BLOCK_SIZE;
            int col = i % BLOCK_SIZE;
            dia[row][col] = m[offset * matrix_dim + offset + row * matrix_dim + col];
        }

        // Load peri-row elements
        for (i = threadIdx.x; i < BLOCK_SIZE * BLOCK_SIZE; i += blockDim.x) {
            int row = i / BLOCK_SIZE;
            int col = i % BLOCK_SIZE;
            peri_row[row][col] = m[offset * matrix_dim + offset + row * matrix_dim + 
                                 (blockIdx.x + 1) * BLOCK_SIZE + col];
        }
    } else {
        idx = threadIdx.x - BLOCK_SIZE;
        // Load second half of diagonal
        for (i = threadIdx.x - BLOCK_SIZE; i < BLOCK_SIZE * BLOCK_SIZE; i += blockDim.x) {
            if (i >= BLOCK_SIZE * BLOCK_SIZE / 2) {
                int row = i / BLOCK_SIZE;
                int col = i % BLOCK_SIZE;
                dia[row][col] = m[offset * matrix_dim + offset + row * matrix_dim + col];
            }
        }

        // Load peri-col elements
        for (i = threadIdx.x - BLOCK_SIZE; i < BLOCK_SIZE * BLOCK_SIZE; i += blockDim.x) {
            int row = i / BLOCK_SIZE;
            int col = i % BLOCK_SIZE;
            peri_col[row][col] = m[(offset + (blockIdx.x + 1) * BLOCK_SIZE + row) * matrix_dim + 
                                  offset + col];
        }
    }
    __syncthreads();

    // Computation phase
    if (threadIdx.x < BLOCK_SIZE) { // peri-row
        idx = threadIdx.x;
        for (i = 1; i < BLOCK_SIZE; i++) {
            for (j = 0; j < i; j++)
                peri_row[i][idx] -= dia[i][j] * peri_row[j][idx];
        }
    } else { // peri-col
        idx = threadIdx.x - BLOCK_SIZE;
        for (i = 0; i < BLOCK_SIZE; i++) {
            float sum = 0.0f;
            for (j = 0; j < i; j++)
                sum += peri_col[idx][j] * dia[j][i];
            peri_col[idx][i] -= sum;
            peri_col[idx][i] /= dia[i][i];
        }
    }
    __syncthreads();

    // Coalesced writing back to global memory
    if (threadIdx.x < BLOCK_SIZE) { // peri-row
        for (i = 1 + threadIdx.x; i < BLOCK_SIZE; i += blockDim.x) {
            int row = i - 1;
            int col = threadIdx.x;
            m[(offset + 1 + row) * matrix_dim + offset + (blockIdx.x + 1) * BLOCK_SIZE + col] =
                peri_row[i][threadIdx.x];
        }
    } else { // peri-col
        for (i = threadIdx.x - BLOCK_SIZE; i < BLOCK_SIZE * BLOCK_SIZE; i += blockDim.x) {
            int row = i / BLOCK_SIZE;
            int col = i % BLOCK_SIZE;
            m[(offset + (blockIdx.x + 1) * BLOCK_SIZE + row) * matrix_dim + offset + col] = 
                peri_col[row][col];
        }
    }
}

__global__ void lud_internal(float *m, int matrix_dim, int offset) {
    __shared__ float peri_row[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float peri_col[BLOCK_SIZE][BLOCK_SIZE];

    int i;
    float sum;

    int global_row_id = offset + (blockIdx.y + 1) * BLOCK_SIZE;
    int global_col_id = offset + (blockIdx.x + 1) * BLOCK_SIZE;

    // Coalesced loading of shared memory
    int row_idx = threadIdx.y;
    int col_idx = threadIdx.x;
    
    // Load peri_row with coalesced access
    peri_row[row_idx][col_idx] = 
        m[(offset + row_idx) * matrix_dim + global_col_id + col_idx];
    
    // Load peri_col with coalesced access
    peri_col[row_idx][col_idx] = 
        m[(global_row_id + row_idx) * matrix_dim + offset + col_idx];

    __syncthreads();

    // Computation using all threads in block
    sum = 0;
    for (i = 0; i < BLOCK_SIZE; i++)
        sum += peri_col[row_idx][i] * peri_row[i][col_idx];
    
    // Write back with coalesced access
    m[(global_row_id + row_idx) * matrix_dim + global_col_id + col_idx] -= sum;
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
