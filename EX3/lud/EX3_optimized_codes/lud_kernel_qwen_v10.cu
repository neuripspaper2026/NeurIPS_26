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
        float4* src = (float4*)&m[array_offset + tid];
        float4* dst = (float4*)&shadow[tid][0];
        #pragma unroll
        for (i = 0; i < BLOCK_SIZE; i++) {
            dst[i] = src[i * (matrix_dim/4)];
        }
    }
    __syncthreads();
    
    for (i = 0; i < BLOCK_SIZE - 1; i++) {
        if (tid > i) {
            #pragma unroll
            for (j = 0; j < i; j++)
                shadow[tid][i] -= shadow[tid][j] * shadow[j][i];
            shadow[tid][i] /= shadow[i][i];
        }

        __syncthreads();
        
        if (tid > i) {
            #pragma unroll
            for (j = 0; j < i + 1; j++)
                shadow[i + 1][tid] -=
                    shadow[i + 1][j] * shadow[j][tid];
        }
        __syncthreads();
    }

    array_offset = (offset + 1) * matrix_dim + offset;
    if (tid < BLOCK_SIZE) {
        float4* src = (float4*)&shadow[1][tid];
        float4* dst = (float4*)&m[array_offset + tid];
        #pragma unroll
        for (i = 0; i < BLOCK_SIZE - 1; i++) {
            dst[i * (matrix_dim/4)] = src[i];
        }
    }
}

__global__ void lud_perimeter(float *m, int matrix_dim, int offset) {
    __shared__ float dia[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float peri_row[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float peri_col[BLOCK_SIZE][BLOCK_SIZE];

    int i, j, array_offset;
    int idx;

    if (threadIdx.x < BLOCK_SIZE) {
        idx = threadIdx.x;

        array_offset = offset * matrix_dim + offset;
        float4* src_dia = (float4*)&m[array_offset + idx];
        float4* dst_dia = (float4*)&dia[0][idx];
        #pragma unroll
        for (i = 0; i < BLOCK_SIZE/2; i++) {
            dst_dia[i] = src_dia[i * (matrix_dim/4)];
        }

        array_offset = offset * matrix_dim + offset + (blockIdx.x + 1) * BLOCK_SIZE;
        float4* src_row = (float4*)&m[array_offset + idx];
        float4* dst_row = (float4*)&peri_row[0][idx];
        #pragma unroll
        for (i = 0; i < BLOCK_SIZE/4; i++) {
            dst_row[i] = src_row[i * (matrix_dim/4)];
        }

    } else {
        idx = threadIdx.x - BLOCK_SIZE;

        array_offset = (offset + BLOCK_SIZE / 2) * matrix_dim + offset;
        float4* src_dia = (float4*)&m[array_offset + idx];
        float4* dst_dia = (float4*)&dia[BLOCK_SIZE/2][idx];
        #pragma unroll
        for (i = 0; i < BLOCK_SIZE/2; i++) {
            dst_dia[i] = src_dia[i * (matrix_dim/4)];
        }

        array_offset = (offset + (blockIdx.x + 1) * BLOCK_SIZE) * matrix_dim + offset;
        float4* src_col = (float4*)&m[array_offset + idx];
        float4* dst_col = (float4*)&peri_col[0][idx];
        #pragma unroll
        for (i = 0; i < BLOCK_SIZE/4; i++) {
            dst_col[i] = src_col[i * (matrix_dim/4)];
        }
    }
    __syncthreads();

    if (threadIdx.x < BLOCK_SIZE) { // peri-row
        idx = threadIdx.x;
        #pragma unroll
        for (i = 1; i < BLOCK_SIZE; i++) {
            #pragma unroll
            for (j = 0; j < i; j++)
                peri_row[i][idx] -= dia[i][j] * peri_row[j][idx];
        }
    } else { // peri-col
        idx = threadIdx.x - BLOCK_SIZE;
        #pragma unroll
        for (i = 0; i < BLOCK_SIZE; i++) {
            #pragma unroll
            for (j = 0; j < i; j++)
                peri_col[idx][i] -= peri_col[idx][j] * dia[j][i];
            peri_col[idx][i] /= dia[i][i];
        }
    }

    __syncthreads();

    if (threadIdx.x < BLOCK_SIZE) { // peri-row
        idx = threadIdx.x;
        array_offset = (offset + 1) * matrix_dim + offset + (blockIdx.x + 1) * BLOCK_SIZE;
        float4* src = (float4*)&peri_row[1][idx];
        float4* dst = (float4*)&m[array_offset + idx];
        #pragma unroll
        for (i = 0; i < (BLOCK_SIZE-1)/4; i++) {
            dst[i * (matrix_dim/4)] = src[i];
        }
    } else { // peri-col
        idx = threadIdx.x - BLOCK_SIZE;
        array_offset = (offset + (blockIdx.x + 1) * BLOCK_SIZE) * matrix_dim + offset;
        float4* src = (float4*)&peri_col[0][idx];
        float4* dst = (float4*)&m[array_offset + idx];
        #pragma unroll
        for (i = 0; i < BLOCK_SIZE/4; i++) {
            dst[i * (matrix_dim/4)] = src[i];
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

    // Coalesced memory access using float4
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (tid < BLOCK_SIZE * BLOCK_SIZE) {
        int row = tid / BLOCK_SIZE;
        int col = tid % BLOCK_SIZE;
        peri_row[row][col] = m[(offset + row) * matrix_dim + global_col_id + col];
        peri_col[row][col] = m[(global_row_id + row) * matrix_dim + offset + col];
    }
    
    __syncthreads();

    sum = 0;
    #pragma unroll 8
    for (i = 0; i < BLOCK_SIZE; i++)
        sum += peri_col[threadIdx.y][i] * peri_row[i][threadIdx.x];
    
    m[(global_row_id + threadIdx.y) * matrix_dim + global_col_id + threadIdx.x] -= sum;
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
