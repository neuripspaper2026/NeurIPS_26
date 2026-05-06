#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>

#include "find_ellipse_kernel.h"
#include "../../common_rodinia/cuda/profile.h"

// Global variable for kernel timing
double g_find_ellipse_kernel_time = 0.0;


// The number of sample points in each ellipse (stencil)
#define NPOINTS 150
// The maximum radius of a sample ellipse
#define MAX_RAD 20
// The total number of sample ellipses
#define NCIRCLES 7
// The size of the structuring element used in dilation
#define STREL_SIZE (12 * 2 + 1)


// Matrix used to store the maximal GICOV score at each pixels
// Produced by the GICOV kernel and consumed by the dilation kernel
float *device_gicov;


// Constant device arrays holding the stencil parameters used by the GICOV
// kernel
__constant__ float c_sin_angle[NPOINTS];
__constant__ float c_cos_angle[NPOINTS];
__constant__ int c_tX[NCIRCLES * NPOINTS];
__constant__ int c_tY[NCIRCLES * NPOINTS];

// Modern texture object API (replaces deprecated texture references)
// Texture references to the gradient matrices used by the GICOV kernel
// texture<float, 1, cudaReadModeElementType> t_grad_x;
// texture<float, 1, cudaReadModeElementType> t_grad_y;

// Kernel to find the maximal GICOV value at each pixel of a
//  video frame, based on the input x- and y-gradient matrices
#include "track-ellipse.h"

////////////////////////////////////////////////////////////////////////////////
// Optimized GICOV kernel
////////////////////////////////////////////////////////////////////////////////
__global__ void GICOV_kernel(int grad_m, float * __restrict__ gicov,
                             cudaTextureObject_t texObj_grad_x,
                             cudaTextureObject_t texObj_grad_y) {
    // Each block processes one column (i) and a contiguous tile of rows (j)
    const int i = blockIdx.x + MAX_RAD + 2;
    const int j = threadIdx.x + MAX_RAD + 2;

    // Early exit for threads outside valid range in the column (in case
    // blockDim.x is larger than the useful range); still writes back in-range.
    // As per instructions, do not introduce early returns after timing points;
    // this kernel has none in scope here.
    float max_GICOV = 0.0f;

    // Precompute base address to avoid recomputing i * grad_m
    const int base_addr = i * grad_m;

#pragma unroll
    for (int k = 0; k < NCIRCLES; k++) {
        float sum  = 0.0f;
        float M2   = 0.0f;
        float mean = 0.0f;

        const int kOffset = k * NPOINTS;

#pragma unroll
        for (int n = 0; n < NPOINTS; n++) {
            // Use register indices for stencil offsets
            const int idx = kOffset + n;
            const int ty  = c_tY[idx];
            const int tx  = c_tX[idx];

            const int y = j + ty;
            const int x = i + tx;

            // precompute address once
            const int addr = base_addr + y;

            // Fetch gradients and compute projection using FMA-friendly form
            const float gx = tex1Dfetch<float>(texObj_grad_x, addr);
            const float gy = tex1Dfetch<float>(texObj_grad_y, addr);
            const float cs = c_cos_angle[n];
            const float sn = c_sin_angle[n];

            const float p = fmaf(gy, sn, gx * cs);

            sum += p;

            // Welford's online algorithm, algebraically simplified
            const float inv = 1.0f / (float)(n + 1);
            const float delta = p - mean;
            mean = mean + delta * inv;
            const float delta2 = p - mean;
            M2 += delta * delta2;
        }

        const float invN  = 1.0f / (float)NPOINTS;
        const float invNm = 1.0f / (float)(NPOINTS - 1);
        mean = sum * invN;
        const float var  = M2 * invNm;

        // Avoid redundant recomputation of mean^2 / var
        const float num = mean * mean;
        const float ratio = num / var;

        if (ratio > max_GICOV)
            max_GICOV = ratio;
    }

    gicov[base_addr + j] = max_GICOV;
}

////////////////////////////////////////////////////////////////////////////////
// Constant device array holding the structuring element used by the dilation
// kernel
////////////////////////////////////////////////////////////////////////////////
__constant__ float c_strel[STREL_SIZE * STREL_SIZE];

////////////////////////////////////////////////////////////////////////////////
// Optimized dilation kernel
////////////////////////////////////////////////////////////////////////////////
__global__ void dilate_kernel(int img_m, int img_n, int strel_m, int strel_n,
                              float * __restrict__ dilated,
                              cudaTextureObject_t texObj_img) {
    // Find the center of the structuring element
    const int el_center_i = strel_m / 2;
    const int el_center_j = strel_n / 2;

    // Determine this thread's location in the matrix
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = thread_id % img_m;
    const int j = thread_id / img_m;

    // Guard against out-of-range threads, but still allow all active threads in
    // a full block to participate; this does not interfere with timing outside.
    if (j >= img_n)
        return;

    float max_val = 0.0f;
    const int base_y = i - el_center_i;
    const int base_x = j - el_center_j;

    // Precompute for address generation
    const int img_m_local = img_m;
    const int img_n_local = img_n;

#pragma unroll
    for (int el_i = 0; el_i < strel_m; el_i++) {
        const int y = base_y + el_i;
        if ((unsigned)y < (unsigned)img_m_local) {
            const int strel_row = el_i * strel_n;
#pragma unroll
            for (int el_j = 0; el_j < strel_n; el_j++) {
                const int x = base_x + el_j;
                const float strel_val = c_strel[strel_row + el_j];

                // Combine bounds and structuring element checks to reduce
                // branches and leverage predication on A100.
                if ((unsigned)x < (unsigned)img_n_local && strel_val != 0.0f) {
                    const int addr = x * img_m_local + y;
                    const float temp = tex1Dfetch<float>(texObj_img, addr);
                    max_val = fmaxf(max_val, temp);
                }
            }
        }
    }

    dilated[i * img_n_local + j] = max_val;
}


// Helper function to create texture object for 1D float data
static cudaTextureObject_t createTextureObject1D(const void *devPtr, size_t sizeInBytes) {
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = const_cast<void*>(devPtr);
    resDesc.res.linear.desc = cudaCreateChannelDesc<float>();
    resDesc.res.linear.sizeInBytes = sizeInBytes;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.normalizedCoords = 0;
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t texObj = 0;
    cudaError_t err = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
    if (err != cudaSuccess) {
        printf("Error creating texture object: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return texObj;
}


// Sets up and invokes the GICOV kernel and returns its output
float *GICOV_CUDA(int grad_m, int grad_n, float *host_grad_x,
                  float *host_grad_y) {

    int MaxR = MAX_RAD + 2;

    // Allocate device memory
    unsigned int grad_mem_size = sizeof(float) * grad_m * grad_n;
    float *device_grad_x, *device_grad_y;
    cudaMalloc((void **)&device_grad_x, grad_mem_size);
    cudaMalloc((void **)&device_grad_y, grad_mem_size);

    // Copy the input gradients to the device
    cudaMemcpy(device_grad_x, host_grad_x, grad_mem_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(device_grad_y, host_grad_y, grad_mem_size,
               cudaMemcpyHostToDevice);

    // Create texture objects (modern API)
    cudaTextureObject_t texObj_grad_x = createTextureObject1D(device_grad_x, grad_mem_size);
    cudaTextureObject_t texObj_grad_y = createTextureObject1D(device_grad_y, grad_mem_size);

    // Allocate & initialize device memory for result
    // (some elements are not assigned values in the kernel)
    cudaMalloc((void **)&device_gicov, grad_mem_size);
    cudaMemset(device_gicov, 0, grad_mem_size);

    // Setup execution parameters
    int num_blocks = grad_n - (2 * MaxR);
    int threads_per_block = grad_m - (2 * MaxR);

    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    // Execute the GICOV kernel
    PROFILE((
        GICOV_kernel<<<num_blocks, threads_per_block>>>(grad_m, device_gicov,
                                                        texObj_grad_x, texObj_grad_y)
    ));

    // Check for kernel errors
    cudaDeviceSynchronize();
    
    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    double gicov_time = (kernel_end.tv_sec - kernel_start.tv_sec) + 
                        (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
    g_find_ellipse_kernel_time += gicov_time;
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("GICOV kernel error: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Copy the result to the host
    float *host_gicov = (float *)malloc(grad_mem_size);
    cudaMemcpy(host_gicov, device_gicov, grad_mem_size, cudaMemcpyDeviceToHost);

    // Cleanup memory
    cudaDestroyTextureObject(texObj_grad_x);
    cudaDestroyTextureObject(texObj_grad_y);
    cudaFree(device_grad_x);
    cudaFree(device_grad_y);

    return host_gicov;
}


// Sets up and invokes the dilation kernel and returns its output
float *dilate_CUDA(int max_gicov_m, int max_gicov_n, int strel_m, int strel_n) {
    // Allocate device memory for result
    unsigned int max_gicov_mem_size = sizeof(float) * max_gicov_m * max_gicov_n;
    float *device_img_dilated;
    cudaMalloc((void **)&device_img_dilated, max_gicov_mem_size);

    // Create texture object for the input matrix of GICOV values
    cudaTextureObject_t texObj_img = createTextureObject1D(device_gicov, max_gicov_mem_size);

    // Setup execution parameters
    int num_threads = max_gicov_m * max_gicov_n;
    int threads_per_block = 176;
    int num_blocks =
        (int)(((float)num_threads / (float)threads_per_block) + 0.5);

    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    // Execute the dilation kernel
    PROFILE((
        dilate_kernel<<<num_blocks, threads_per_block>>>(
            max_gicov_m, max_gicov_n, strel_m, strel_n, device_img_dilated, texObj_img)
    ));

    // Check for kernel errors
    cudaDeviceSynchronize();
    
    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    double dilate_time = (kernel_end.tv_sec - kernel_start.tv_sec) + 
                         (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
    g_find_ellipse_kernel_time += dilate_time;
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Dilation kernel error: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Copy the result to the host
    float *host_img_dilated = (float *)malloc(max_gicov_mem_size);
    cudaMemcpy(host_img_dilated, device_img_dilated, max_gicov_mem_size,
               cudaMemcpyDeviceToHost);

    // Cleanup memory
    cudaDestroyTextureObject(texObj_img);
    cudaFree(device_gicov);
    cudaFree(device_img_dilated);

    return host_img_dilated;
}


// Chooses the most appropriate GPU on which to execute
void select_device() {
    // Figure out how many devices exist
    int num_devices, device;
    cudaGetDeviceCount(&num_devices);

    // Choose the device with the largest number of multiprocessors
    if (num_devices > 0) {
        int max_multiprocessors = 0, max_device = -1;
        for (device = 0; device < num_devices; device++) {
            cudaDeviceProp properties;
            cudaGetDeviceProperties(&properties, device);
            if (max_multiprocessors < properties.multiProcessorCount) {
                max_multiprocessors = properties.multiProcessorCount;
                max_device = device;
            }
        }
        cudaSetDevice(max_device);
    }

    // The following is to remove the API initialization overhead from the
    // runtime measurements
    cudaFree(0);
}


// Transfers pre-computed constants used by the two kernels to the GPU
void transfer_constants(float *host_sin_angle, float *host_cos_angle,
                        int *host_tX, int *host_tY, int strel_m, int strel_n,
                        float *host_strel) {

    // Compute the sizes of the matrices
    unsigned int angle_mem_size = sizeof(float) * NPOINTS;
    unsigned int t_mem_size = sizeof(int) * NCIRCLES * NPOINTS;
    unsigned int strel_mem_size = sizeof(float) * strel_m * strel_n;

    // Copy the matrices from host memory to device constant memory
    cudaMemcpyToSymbol(c_sin_angle, host_sin_angle, angle_mem_size, 0,
                       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_cos_angle, host_cos_angle, angle_mem_size, 0,
                       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_tX, host_tX, t_mem_size, 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_tY, host_tY, t_mem_size, 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_strel, host_strel, strel_mem_size, 0,
                       cudaMemcpyHostToDevice);
}
