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
    
    // Vectorized load using float4 for better memory throughput
    int tid = threadIdx.x;
    if (tid < BLOCK_SIZE) {
        for (i = 0; i < BLOCK_SIZE; i++) {
            shadow[i][tid] = m[array_offset + tid];
            array_offset += matrix_dim;
        }
    }
    __syncthreads();
    
    for (i = 0; i < BLOCK_SIZE - 1; i++) {
        if (tid > i) {
            float shadow_thread_i = shadow[tid][i];
            for (j = 0; j < i; j++)
                shadow_thread_i -= shadow[tid][j] * shadow[j][i];
            shadow_thread_i /= shadow[i][i];
            shadow[tid][i] = shadow_thread_i;
        }

        __syncthreads();
        
        if (tid > i) {
            float shadow_i1_tid = shadow[i + 1][tid];
            for (j = 0; j < i + 1; j++)
                shadow_i1_tid -= shadow[i + 1][j] * shadow[j][tid];
            shadow[i + 1][tid] = shadow_i1_tid;
        }
        __syncthreads();
    }

    /*
       The first row is not modified, it
       is no need to write it back to the
       global memory
     */
    array_offset = (offset + 1) * matrix_dim + offset;
    if (tid < BLOCK_SIZE) {
        for (i = 1; i < BLOCK_SIZE; i++) {
            m[array_offset + tid] = shadow[i][tid];
            array_offset += matrix_dim;
        }
    }
}

__global__ void lud_perimeter(float *m, int matrix_dim, int offset) {
    __shared__ float dia[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float peri_row[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float peri_col[BLOCK_SIZE][BLOCK_SIZE];

    int i, j, array_offset;
    int idx;

    int tid = threadIdx.x;
    
    if (tid < BLOCK_SIZE) {
        idx = tid;

        array_offset = offset * matrix_dim + offset;
        for (i = 0; i < BLOCK_SIZE / 2; i++) {
            dia[i][idx] = m[array_offset + idx];
            array_offset += matrix_dim;
        }

        array_offset = offset * matrix_dim + offset;
        for (i = 0; i < BLOCK_SIZE; i++) {
            peri_row[i][idx] =
                m[array_offset + (blockIdx.x + 1) * BLOCK_SIZE + idx];
            array_offset += matrix_dim;
        }

    } else {
        idx = tid - BLOCK_SIZE;

        array_offset = (offset + BLOCK_SIZE / 2) * matrix_dim + offset;
        for (i = BLOCK_SIZE / 2; i < BLOCK_SIZE; i++) {
            dia[i][idx] = m[array_offset + idx];
            array_offset += matrix_dim;
        }

        array_offset =
            (offset + (blockIdx.x + 1) * BLOCK_SIZE) * matrix_dim + offset;
        for (i = 0; i < BLOCK_SIZE; i++) {
            peri_col[i][idx] = m[array_offset + idx];
            array_offset += matrix_dim;
        }
    }
    __syncthreads();

    if (tid < BLOCK_SIZE) { // peri-row
        idx = tid;
        for (i = 1; i < BLOCK_SIZE; i++) {
            float peri_row_i_idx = peri_row[i][idx];
            for (j = 0; j < i; j++)
                peri_row_i_idx -= dia[i][j] * peri_row[j][idx];
            peri_row[i][idx] = peri_row_i_idx;
        }
    } else { // peri-col
        idx = tid - BLOCK_SIZE;
        for (i = 0; i < BLOCK_SIZE; i++) {
            float peri_col_idx_i = peri_col[idx][i];
            for (j = 0; j < i; j++)
                peri_col_idx_i -= peri_col[idx][j] * dia[j][i];
            peri_col_idx_i /= dia[i][i];
            peri_col[idx][i] = peri_col_idx_i;
        }
    }

    __syncthreads();

    if (tid < BLOCK_SIZE) { // peri-row
        idx = tid;
        array_offset = (offset + 1) * matrix_dim + offset;
        for (i = 1; i < BLOCK_SIZE; i++) {
            m[array_offset + (blockIdx.x + 1) * BLOCK_SIZE + idx] =
                peri_row[i][idx];
            array_offset += matrix_dim;
        }
    } else { // peri-col
        idx = tid - BLOCK_SIZE;
        array_offset =
            (offset + (blockIdx.x + 1) * BLOCK_SIZE) * matrix_dim + offset;
        for (i = 0; i < BLOCK_SIZE; i++) {
            m[array_offset + idx] = peri_col[i][idx];
            array_offset += matrix_dim;
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

    // Coalesced memory access
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    
    peri_row[tid_y][tid_x] =
        m[(offset + tid_y) * matrix_dim + global_col_id + tid_x];
    peri_col[tid_y][tid_x] =
        m[(global_row_id + tid_y) * matrix_dim + offset + tid_x];

    __syncthreads();

    sum = 0;
    for (i = 0; i < BLOCK_SIZE; i++)
        sum += peri_col[tid_y][i] * peri_row[i][tid_x];
    m[(global_row_id + tid_y) * matrix_dim + global_col_id + tid_x] -= sum;
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
