#include "misc_math.h"
#include "track_ellipse_kernel.h"
#include "../../common_rodinia/cuda/profile.h"
#include <time.h>

// Global variable for kernel timing
double g_track_ellipse_kernel_time = 0.0;

// Constants used in the MGVF computation
#define ONE_OVER_PI (1.0 / PI)
#define MU 0.5
#define LAMBDA (8.0 * MU + 1.0)


// Host and device arrays to hold device pointers to input matrices
float **host_I_array, **host_IMGVF_array;
float **device_I_array, **device_IMGVF_array;
// Host and device arrays to hold sizes of input matrices
int *host_m_array, *host_n_array;
int *device_m_array, *device_n_array;

// Host array to hold matrices for all cells
// (so we can copy to and from the device in a single transfer)
float *host_I_all;
int total_mem_size;

// The number of threads per thread block
const int threads_per_block = 320;
// next_lowest_power_of_two = 2^(floor(log2(threads_per_block)))
const int next_lowest_power_of_two = 256;


// Regularized version of the Heaviside step function:
// He(x) = (atan(x) / pi) + 0.5
__device__ float heaviside(float x) {
    return (atan(x) * ONE_OVER_PI) + 0.5;

    // A simpler, faster approximation of the Heaviside function
    /* float out = 0.0;
    if (x > -0.0001) out = 0.5;
    if (x >  0.0001) out = 1.0;
    return out; */
}


// Kernel to compute the Motion Gradient Vector Field (MGVF) matrix for multiple
// cells
#include "track-ellipse.h"

__global__ void IMGVF_kernel(float **IMGVF_array, float **I_array, int *m_array,
                             int *n_array, float vx, float vy, float e,
                             int max_iterations, float cutoff) {

    __shared__ float IMGVF[41 * 81];

    __shared__ float buffer[threads_per_block];

    int cell_num = blockIdx.x;

    float *IMGVF_global = IMGVF_array[cell_num];
    float *I = I_array[cell_num];

    int m = m_array[cell_num];
    int n = n_array[cell_num];

    const int thread_id = threadIdx.x;

    const int num_elems = m * n;
    const int max = (num_elems + threads_per_block - 1) / threads_per_block;

    // Cache 1/n as float for use in integer division via float multiply
    const float inv_n = 1.0f / (float)n;

    // Load initial IMGVF into shared memory with coalesced access
    for (int t = thread_id; t < num_elems; t += threads_per_block) {
        IMGVF[t] = IMGVF_global[t];
    }
    __syncthreads();

    __shared__ int cell_converged;
    if (thread_id == 0)
        cell_converged = 0;
    __syncthreads();

    const int tid_mod = thread_id % n;
    const int tbsize_mod = threads_per_block % n;

    const float one_over_e = 1.0f / e;

    int iterations = 0;

    // Precompute some constants used inside the loop
    const float mu_over_lambda = MU / LAMBDA;
    const float inv_lambda = 1.0f / LAMBDA;

    while ((!cell_converged) && (iterations < max_iterations)) {

        float total_diff = 0.0f;

        int i = 0, j = 0;
        int old_i = 0, old_j = 0;
        j = tid_mod - tbsize_mod;

#pragma unroll 1
        for (int thread_block = 0; thread_block < max; thread_block++) {

            old_i = i;
            old_j = j;

            int offset = thread_block * threads_per_block;
            int idx = thread_id + offset;
            i = (int)((float)idx * inv_n);
            j += tbsize_mod;
            if (j >= n)
                j -= n;

            float new_val = 0.0f;
            float old_val = 0.0f;

            if (i < m) {
                int rowU = (i == 0) ? 0 : i - 1;
                int rowD = (i == m - 1) ? m - 1 : i + 1;
                int colL = (j == 0) ? 0 : j - 1;
                int colR = (j == n - 1) ? n - 1 : j + 1;

                int base = i * n;
                int baseU = rowU * n;
                int baseD = rowD * n;

                int idxC = base + j;
                int idxL = base + colL;
                int idxR = base + colR;
                int idxU = baseU + j;
                int idxD = baseD + j;
                int idxUL = baseU + colL;
                int idxUR = baseU + colR;
                int idxDL = baseD + colL;
                int idxDR = baseD + colR;

                old_val = IMGVF[idxC];

                float U = IMGVF[idxU] - old_val;
                float D = IMGVF[idxD] - old_val;
                float L = IMGVF[idxL] - old_val;
                float R = IMGVF[idxR] - old_val;
                float UR = IMGVF[idxUR] - old_val;
                float DR = IMGVF[idxDR] - old_val;
                float UL = IMGVF[idxUL] - old_val;
                float DL = IMGVF[idxDL] - old_val;

                float UHe  = heaviside((U  * -vy)        * one_over_e);
                float DHe  = heaviside((D  *  vy)        * one_over_e);
                float LHe  = heaviside((L  * -vx)        * one_over_e);
                float RHe  = heaviside((R  *  vx)        * one_over_e);
                float URHe = heaviside((UR * (vx - vy))  * one_over_e);
                float DRHe = heaviside((DR * (vx + vy))  * one_over_e);
                float ULHe = heaviside((UL * (-vx - vy)) * one_over_e);
                float DLHe = heaviside((DL * (-vx + vy)) * one_over_e);

                float lap = fmaf(UHe, U, 0.0f);
                lap = fmaf(DHe, D, lap);
                lap = fmaf(LHe, L, lap);
                lap = fmaf(RHe, R, lap);
                lap = fmaf(URHe, UR, lap);
                lap = fmaf(DRHe, DR, lap);
                lap = fmaf(ULHe, UL, lap);
                lap = fmaf(DLHe, DL, lap);

                new_val = fmaf(mu_over_lambda, lap, old_val);

                float vI = I[idxC];
                float diff = new_val - vI;
                new_val = fmaf(-inv_lambda * vI, diff, new_val);
            }

            __syncthreads();

            if (thread_block > 0) {
                if (old_i < m)
                    IMGVF[old_i * n + old_j] = buffer[thread_id];
            }

            if (thread_block < max - 1) {
                buffer[thread_id] = new_val;
            } else {
                if (i < m)
                    IMGVF[i * n + j] = new_val;
            }

            total_diff += fabsf(new_val - old_val);

            __syncthreads();
        }

        buffer[thread_id] = total_diff;
        __syncthreads();

        if (thread_id >= next_lowest_power_of_two) {
            buffer[thread_id - next_lowest_power_of_two] += buffer[thread_id];
        }
        __syncthreads();

        for (int th = next_lowest_power_of_two >> 1; th > 0; th >>= 1) {
            if (thread_id < th) {
                buffer[thread_id] += buffer[thread_id + th];
            }
            __syncthreads();
        }

        if (thread_id == 0) {
            float mean = buffer[0] / (float)(m * n);
            if (mean < cutoff) {
                cell_converged = 1;
            }
        }
        __syncthreads();

        iterations++;
    }

    // Store final IMGVF back to global memory with coalesced access
    for (int t = thread_id; t < num_elems; t += threads_per_block) {
        IMGVF_global[t] = IMGVF[t];
    }
}


// Host function that launches a CUDA kernel to compute the MGVF matrices for
// the specified cells
void IMGVF_cuda(MAT **I, MAT **IMGVF, double vx, double vy, double e,
                int max_iterations, double cutoff, int num_cells) {

    // Initialize the data on the GPU
    IMGVF_cuda_init(I, num_cells);

    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    // Compute the MGVF on the GPU
    PROFILE((
        IMGVF_kernel<<<num_cells, threads_per_block>>>(
            device_IMGVF_array, device_I_array, device_m_array, device_n_array,
            (float)vx, (float)vy, (float)e, max_iterations, (float)cutoff)
    ));

    // Check for kernel errors
    cudaDeviceSynchronize();
    
    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    g_track_ellipse_kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) + 
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("MGVF kernel error: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Copy back the final results from the GPU
    IMGVF_cuda_cleanup(IMGVF, num_cells);
}


// Initializes data on the GPU for the MGVF kernel
void IMGVF_cuda_init(MAT **IE, int num_cells) {
    // Allocate arrays of pointers to device memory
    host_I_array = (float **)malloc(sizeof(float *) * num_cells);
    host_IMGVF_array = (float **)malloc(sizeof(float *) * num_cells);
    cudaMalloc((void **)&device_I_array, num_cells * sizeof(float *));
    cudaMalloc((void **)&device_IMGVF_array, num_cells * sizeof(float *));

    // Allocate arrays of memory dimensions
    host_m_array = (int *)malloc(sizeof(int) * num_cells);
    host_n_array = (int *)malloc(sizeof(int) * num_cells);
    cudaMalloc((void **)&device_m_array, num_cells * sizeof(int));
    cudaMalloc((void **)&device_n_array, num_cells * sizeof(int));

    // Figure out the size of all of the matrices combined
    int i, j, cell_num;
    int total_size = 0;
    for (cell_num = 0; cell_num < num_cells; cell_num++) {
        MAT *I = IE[cell_num];
        int size = I->m * I->n;
        total_size += size;
    }
    total_mem_size = total_size * sizeof(float);

    // Allocate host memory just once for all cells
    host_I_all = (float *)malloc(total_mem_size);

    // Allocate device memory just once for all cells
    float *device_I_all, *device_IMGVF_all;
    cudaMalloc((void **)&device_I_all, total_mem_size);
    cudaMalloc((void **)&device_IMGVF_all, total_mem_size);

    // Copy each initial matrix into the allocated host memory
    int offset = 0;
    for (cell_num = 0; cell_num < num_cells; cell_num++) {
        MAT *I = IE[cell_num];

        // Determine the size of the matrix
        int m = I->m, n = I->n;
        int size = m * n;

        // Store memory dimensions
        host_m_array[cell_num] = m;
        host_n_array[cell_num] = n;

        // Store pointers to allocated memory
        float *device_I = &(device_I_all[offset]);
        float *device_IMGVF = &(device_IMGVF_all[offset]);
        host_I_array[cell_num] = device_I;
        host_IMGVF_array[cell_num] = device_IMGVF;

        // Copy matrix I (which is also the initial IMGVF matrix) into the
        // overall array
        for (i = 0; i < m; i++)
            for (j = 0; j < n; j++)
                host_I_all[offset + (i * n) + j] = (float)m_get_val(I, i, j);

        offset += size;
    }

    // Copy I matrices (which are also the initial IMGVF matrices) to device
    cudaMemcpy(device_I_all, host_I_all, total_mem_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(device_IMGVF_all, host_I_all, total_mem_size,
               cudaMemcpyHostToDevice);

    // Copy pointer arrays to device
    cudaMemcpy(device_I_array, host_I_array, num_cells * sizeof(float *),
               cudaMemcpyHostToDevice);
    cudaMemcpy(device_IMGVF_array, host_IMGVF_array,
               num_cells * sizeof(float *), cudaMemcpyHostToDevice);

    // Copy memory dimension arrays to device
    cudaMemcpy(device_m_array, host_m_array, num_cells * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(device_n_array, host_n_array, num_cells * sizeof(int),
               cudaMemcpyHostToDevice);
}


// Copies the results of the MGVF kernel back to the host
void IMGVF_cuda_cleanup(MAT **IMGVF_out_array, int num_cells) {
    // Copy the result matrices from the device to the host
    cudaMemcpy(host_I_all, host_IMGVF_array[0], total_mem_size,
               cudaMemcpyDeviceToHost);

    // Copy each result matrix into its appropriate host matrix
    int cell_num, offset = 0;
    for (cell_num = 0; cell_num < num_cells; cell_num++) {
        MAT *IMGVF_out = IMGVF_out_array[cell_num];

        // Determine the size of the matrix
        int m = IMGVF_out->m, n = IMGVF_out->n, i, j;
        // Pack the result into the matrix
        for (i = 0; i < m; i++)
            for (j = 0; j < n; j++)
                m_set_val(IMGVF_out, i, j,
                          (double)host_I_all[offset + (i * n) + j]);

        offset += (m * n);
    }

    // Free device memory
    cudaFree(device_m_array);
    cudaFree(device_n_array);
    cudaFree(device_IMGVF_array);
    cudaFree(device_I_array);
    cudaFree(host_IMGVF_array[0]);
    cudaFree(host_I_array[0]);

    // Free host memory
    free(host_m_array);
    free(host_n_array);
    free(host_IMGVF_array);
    free(host_I_array);
    free(host_I_all);
}
