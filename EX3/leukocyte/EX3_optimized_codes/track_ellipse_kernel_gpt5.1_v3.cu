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

__global__ void IMGVF_kernel(float ** __restrict__ IMGVF_array,
                             float ** __restrict__ I_array,
                             const int * __restrict__ m_array,
                             const int * __restrict__ n_array,
                             const float vx, const float vy, const float e,
                             const int max_iterations, const float cutoff) {

    __shared__ float IMGVF[41 * 81];
    __shared__ float buffer[threads_per_block];
    __shared__ int   cell_converged;

    const int cell_num  = blockIdx.x;
    float * __restrict__ IMGVF_global = IMGVF_array[cell_num];
    float * __restrict__ I            = I_array[cell_num];

    const int m = m_array[cell_num];
    const int n = n_array[cell_num];

    const int thread_id = threadIdx.x;
    const int max       = (m * n + threads_per_block - 1) / threads_per_block;

    // preload IMGVF into shared memory
    for (int thread_block = 0; thread_block < max; ++thread_block) {
        const int offset = thread_block * threads_per_block;
        const int idx    = thread_id + offset;
        const int i      = idx / n;
        const int j      = idx - i * n;
        if (i < m) {
            IMGVF[i * n + j] = IMGVF_global[i * n + j];
        }
    }
    __syncthreads();

    if (thread_id == 0)
        cell_converged = 0;
    __syncthreads();

    const float one_nth   = 1.0f / (float)n;
    const int   tid_mod   = thread_id % n;
    const int   tbsize_mod = threads_per_block % n;
    const float one_over_e = 1.0f / e;

    int iterations = 0;

    while ((!cell_converged) && (iterations < max_iterations)) {

        float total_diff = 0.0f;

        int i = 0, j = 0;
        int old_i = 0, old_j = 0;
        j = tid_mod - tbsize_mod;

#pragma unroll 1
        for (int thread_block = 0; thread_block < max; ++thread_block) {

            old_i = i;
            old_j = j;

            const int offset = thread_block * threads_per_block;
            i = (int)((thread_id + offset) * one_nth);
            j += tbsize_mod;
            if (j >= n)
                j -= n;

            float new_val = 0.0f;
            float old_val = 0.0f;

            if (i < m) {
                const int base_idx = i * n + j;

                const int rowU = (i == 0)     ? 0     : i - 1;
                const int rowD = (i == m - 1) ? m - 1 : i + 1;
                const int colL = (j == 0)     ? 0     : j - 1;
                const int colR = (j == n - 1) ? n - 1 : j + 1;

                const int idxU  = rowU * n + j;
                const int idxD  = rowD * n + j;
                const int idxL  = i * n + colL;
                const int idxR  = i * n + colR;
                const int idxUR = rowU * n + colR;
                const int idxDR = rowD * n + colR;
                const int idxUL = rowU * n + colL;
                const int idxDL = rowD * n + colL;

                old_val = IMGVF[base_idx];

                const float U  = IMGVF[idxU]  - old_val;
                const float D  = IMGVF[idxD]  - old_val;
                const float L  = IMGVF[idxL]  - old_val;
                const float R  = IMGVF[idxR]  - old_val;
                const float UR = IMGVF[idxUR] - old_val;
                const float DR = IMGVF[idxDR] - old_val;
                const float UL = IMGVF[idxUL] - old_val;
                const float DL = IMGVF[idxDL] - old_val;

                const float UHe  = heaviside((U  * -vy)          * one_over_e);
                const float DHe  = heaviside((D  *  vy)          * one_over_e);
                const float LHe  = heaviside((L  * -vx)          * one_over_e);
                const float RHe  = heaviside((R  *  vx)          * one_over_e);
                const float URHe = heaviside((UR * (vx - vy))    * one_over_e);
                const float DRHe = heaviside((DR * (vx + vy))    * one_over_e);
                const float ULHe = heaviside((UL * (-vx - vy))   * one_over_e);
                const float DLHe = heaviside((DL * (-vx + vy))   * one_over_e);

                const float diff_sum =
                    UHe  * U  + DHe  * D  + LHe  * L  + RHe  * R +
                    URHe * UR + DRHe * DR + ULHe * UL + DLHe * DL;

                new_val = fmaf((MU / LAMBDA), diff_sum, old_val);

                const float vI   = I[base_idx];
                const float tmp  = new_val - vI;
                new_val = fmaf(-(1.0f / LAMBDA) * vI, tmp, new_val);
            }

            __syncthreads();

            if (thread_block > 0) {
                const int prev_offset = (thread_block - 1) * threads_per_block;
                if (old_i < m) {
                    IMGVF[old_i * n + old_j] = buffer[thread_id];
                }
            }

            if (thread_block < max - 1) {
                buffer[thread_id] = new_val;
            } else {
                if (i < m) {
                    IMGVF[i * n + j] = new_val;
                }
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

        int th = next_lowest_power_of_two >> 1;
#pragma unroll
        while (th > 0) {
            if (thread_id < th) {
                buffer[thread_id] += buffer[thread_id + th];
            }
            __syncthreads();
            th >>= 1;
        }

        if (thread_id == 0) {
            const float mean = buffer[0] / (float)(m * n);
            if (mean < cutoff)
                cell_converged = 1;
        }
        __syncthreads();

        ++iterations;
    }

    for (int thread_block = 0; thread_block < max; ++thread_block) {
        const int offset = thread_block * threads_per_block;
        const int idx    = thread_id + offset;
        const int i      = idx / n;
        const int j      = idx - i * n;
        if (i < m) {
            IMGVF_global[i * n + j] = IMGVF[i * n + j];
        }
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
