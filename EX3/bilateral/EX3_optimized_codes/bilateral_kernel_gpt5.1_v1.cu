/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <cuda_runtime.h>
#include <string.h>
#include <texture_types.h>
#include <texture_fetch_functions.h>
#include <helper_math.h>
#include <helper_functions.h>
#include <helper_cuda.h>


////////////////////////////////////////////////////////////////////////////////
//! Device code
////////////////////////////////////////////////////////////////////////////////

__constant__ float cGaussian[64]; // gaussian array in device side

uint *dImage = NULL; // original image
uint *dTemp = NULL; // temp array for iterations
size_t pitch;

/*
    Perform a simple bilateral filter.

    Bilateral filter is a nonlinear filter that is a mixture of range
    filter and domain filter, the previous one preserves crisp edges and
    the latter one filters noise. The intensity value at each pixel in
    an image is replaced by a weighted average of intensity values from
    nearby pixels.

    The weight factor is calculated by the product of domain filter
    component(using the gaussian distribution as a spatial distance) as
    well as range filter component(Euclidean distance between center pixel
    and the current neighbor pixel). Because this process is nonlinear,
    the sample just uses a simple pixel by pixel step.

    Texture fetches automatically clamp to edge of image. 1D gaussian array
    is mapped to a 1D texture instead of using shared memory, which may
    cause severe bank conflict.

    Threads are y-pass(column-pass), because the output is coalesced.

    Parameters
    od - pointer to output data in global memory
    d_f - pointer to the 1D gaussian array
    e_d - euclidean delta
    w  - image width
    h  - image height
    r  - filter radius
*/

// Euclidean Distance (x, y, d) = exp((|x - y| / d)^2 / 2)
__device__ float euclideanLen(float4 a, float4 b, float d) {

    float mod = (b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y) +
                (b.z - a.z) * (b.z - a.z);

    return __expf(-mod / (2.f * d * d));
}

__device__ uint rgbaFloatToInt(float4 rgba) {
    rgba.x = __saturatef(fabs(rgba.x)); // clamp to [0.0, 1.0]
    rgba.y = __saturatef(fabs(rgba.y));
    rgba.z = __saturatef(fabs(rgba.z));
    rgba.w = __saturatef(fabs(rgba.w));
    return (uint(rgba.w * 255.0f) << 24) | (uint(rgba.z * 255.0f) << 16) |
           (uint(rgba.y * 255.0f) << 8) | uint(rgba.x * 255.0f);
}

// column pass using coalesced global memory reads
#include <cuda_runtime.h>

__global__ void d_bilateral_filter(uint *od, int w, int h, float e_d, int r,
                                   cudaTextureObject_t texObj) {
    // Configure block size assumptions for shared memory tiling
    const int BLOCK_W = blockDim.x;
    const int BLOCK_H = blockDim.y;

    int x = blockIdx.x * BLOCK_W + threadIdx.x;
    int y = blockIdx.y * BLOCK_H + threadIdx.y;

    // Shared memory tile for the neighborhood (radius r halo)
    extern __shared__ float4 sTile[];

    const int tileW = BLOCK_W + 2 * r;
    const int tileH = BLOCK_H + 2 * r;

    // Coordinates in shared tile (offset by radius)
    int sx = threadIdx.x + r;
    int sy = threadIdx.y + r;

    // Clamp helper for texture coordinates
    const int maxX = w - 1;
    const int maxY = h - 1;
    int gx = x < 0 ? 0 : (x > maxX ? maxX : x);
    int gy = y < 0 ? 0 : (y > maxY ? maxY : y);

    // Load center pixel for each thread
    if (x < w && y < h) {
        sTile[sy * tileW + sx] = tex2D<float4>(texObj, gx, gy);
    } else {
        // For out-of-range threads (only on fringes of the grid), still load
        // a clamped value to keep shared memory initialized
        sTile[sy * tileW + sx] = tex2D<float4>(texObj, gx, gy);
    }

    // Load halo regions in x-direction
    // Left halo
    for (int dx = threadIdx.x; dx < r * BLOCK_H; dx += BLOCK_W * BLOCK_H) {
        int local_y = dx / r;
        int local_x_off = dx % r;
        int halo_sy = local_y + r;
        int halo_sx = local_x_off;
        int global_x = blockIdx.x * BLOCK_W + halo_sx - r;
        int global_y = blockIdx.y * BLOCK_H + (halo_sy - r);
        int clamped_x = global_x < 0 ? 0 : (global_x > maxX ? maxX : global_x);
        int clamped_y = global_y < 0 ? 0 : (global_y > maxY ? maxY : global_y);
        if (halo_sy < tileH && halo_sx < r) {
            sTile[halo_sy * tileW + halo_sx] =
                tex2D<float4>(texObj, clamped_x, clamped_y);
        }
    }

    // Right halo
    for (int dx = threadIdx.x; dx < r * BLOCK_H; dx += BLOCK_W * BLOCK_H) {
        int local_y = dx / r;
        int local_x_off = dx % r;
        int halo_sy = local_y + r;
        int halo_sx = BLOCK_W + r + local_x_off;
        int global_x = blockIdx.x * BLOCK_W + (halo_sx - r);
        int global_y = blockIdx.y * BLOCK_H + (halo_sy - r);
        int clamped_x = global_x < 0 ? 0 : (global_x > maxX ? maxX : global_x);
        int clamped_y = global_y < 0 ? 0 : (global_y > maxY ? maxY : global_y);
        if (halo_sy < tileH && halo_sx < tileW) {
            sTile[halo_sy * tileW + halo_sx] =
                tex2D<float4>(texObj, clamped_x, clamped_y);
        }
    }

    // Top and bottom halo
    for (int dy = threadIdx.x; dy < BLOCK_W * r; dy += BLOCK_W * BLOCK_H) {
        int local_x = dy / r;
        int local_y_off = dy % r;

        // Top halo
        int halo_sy_top = local_y_off;
        int halo_sx_top = local_x + r;
        int global_x_top = blockIdx.x * BLOCK_W + (halo_sx_top - r);
        int global_y_top = blockIdx.y * BLOCK_H + halo_sy_top - r;
        int clamped_x_top =
            global_x_top < 0 ? 0 : (global_x_top > maxX ? maxX : global_x_top);
        int clamped_y_top =
            global_y_top < 0 ? 0 : (global_y_top > maxY ? maxY : global_y_top);
        if (halo_sy_top < r && halo_sx_top < tileW) {
            sTile[halo_sy_top * tileW + halo_sx_top] =
                tex2D<float4>(texObj, clamped_x_top, clamped_y_top);
        }

        // Bottom halo
        int halo_sy_bot = BLOCK_H + r + local_y_off;
        int halo_sx_bot = local_x + r;
        int global_x_bot = blockIdx.x * BLOCK_W + (halo_sx_bot - r);
        int global_y_bot = blockIdx.y * BLOCK_H + (halo_sy_bot - r);
        int clamped_x_bot =
            global_x_bot < 0 ? 0 : (global_x_bot > maxX ? maxX : global_x_bot);
        int clamped_y_bot =
            global_y_bot < 0 ? 0 : (global_y_bot > maxY ? maxY : global_y_bot);
        if (halo_sy_bot < tileH && halo_sx_bot < tileW) {
            sTile[halo_sy_bot * tileW + halo_sx_bot] =
                tex2D<float4>(texObj, clamped_x_bot, clamped_y_bot);
        }
    }

    __syncthreads();

    if (x >= w || y >= h) {
        return;
    }

    float sum = 0.0f;
    float4 t = {0.f, 0.f, 0.f, 0.f};

    // Center pixel loaded from shared memory
    float4 center = sTile[sy * tileW + sx];

    // Bilateral filter computation using data from shared memory
    #pragma unroll
    for (int i = -r; i <= r; i++) {
        int syi = sy + i;
        #pragma unroll
        for (int j = -r; j <= r; j++) {
            int sxj = sx + j;
            float4 curPix = sTile[syi * tileW + sxj];

            float g_i = cGaussian[i + r];
            float g_j = cGaussian[j + r];

            float diffx = curPix.x - center.x;
            float diffy = curPix.y - center.y;
            float diffz = curPix.z - center.z;

            float mod = diffx * diffx + diffy * diffy + diffz * diffz;
            float range = __expf(-mod / (2.f * e_d * e_d));

            float factor = g_i * g_j * range;

            t.x = fmaf(factor, curPix.x, t.x);
            t.y = fmaf(factor, curPix.y, t.y);
            t.z = fmaf(factor, curPix.z, t.z);
            t.w = fmaf(factor, curPix.w, t.w);

            sum += factor;
        }
    }

    float invSum = 1.0f / sum;
    float4 out;
    out.x = t.x * invSum;
    out.y = t.y * invSum;
    out.z = t.z * invSum;
    out.w = t.w * invSum;

    od[y * w + x] = rgbaFloatToInt(out);
}


////////////////////////////////////////////////////////////////////////////////
//! Host API
////////////////////////////////////////////////////////////////////////////////

extern "C" void initTexture(int width, int height, uint *hImage) {
    // copy image data to array
    checkCudaErrors(
        cudaMallocPitch(&dImage, &pitch, sizeof(uint) * width, height));
    checkCudaErrors(
        cudaMallocPitch(&dTemp, &pitch, sizeof(uint) * width, height));
    checkCudaErrors(cudaMemcpy2D(dImage, pitch, hImage, sizeof(uint) * width,
                                 sizeof(uint) * width, height,
                                 cudaMemcpyHostToDevice));
}

extern "C" void freeTextures() {
    checkCudaErrors(cudaFree(dImage));
    checkCudaErrors(cudaFree(dTemp));
}

/*
    Because a 2D gaussian mask is symmetry in row and column,
    here only generate a 1D mask, and use the product by row
    and column index later.

    1D gaussian distribution :
        g(x, d) -- C * exp(-x^2/d^2), C is a constant amplifier

    parameters:
    og - output gaussian array in global memory
    delta - the 2nd parameter 'd' in the above function
    radius - half of the filter size
             (total filter size = 2 * radius + 1)
*/
extern "C" void updateGaussian(float delta, int radius) {
    float fGaussian[64];

    for (int i = 0; i < 2 * radius + 1; ++i) {
        float x = i - radius;
        fGaussian[i] = expf(-(x * x) / (2 * delta * delta));
    }

    checkCudaErrors(cudaMemcpyToSymbol(cGaussian, fGaussian,
                                       sizeof(float) * (2 * radius + 1)));
}

/*
    Perform 2D bilateral filter on image using CUDA

    Parameters:
    d_dest - pointer to destination image in device memory
    width  - image width
    height - image height
    e_d    - euclidean delta
    radius - filter radius
    iterations - number of iterations

    returns time per iteration
*/

// RGBA version
static cudaTextureObject_t createTextureObject(uint *source, int width,
                                               int height) {
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = source;
    resDesc.res.pitch2D.desc = desc;
    resDesc.res.pitch2D.width = width;
    resDesc.res.pitch2D.height = height;
    resDesc.res.pitch2D.pitchInBytes = pitch;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t texObj = 0;
    checkCudaErrors(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
    return texObj;
}

extern "C" double bilateralFilterRGBA(uint *dDest, int width, int height,
                                      float e_d, int radius, int iterations,
                                      StopWatchInterface *timer) {
    // var for kernel computation timing
    double dKernelTime;

    cudaTextureObject_t texObj = createTextureObject(dImage, width, height);

    for (int i = 0; i < iterations; i++) {
        // sync host and start kernel computation timer
        dKernelTime = 0.0;
        checkCudaErrors(cudaDeviceSynchronize());
        sdkResetTimer(&timer);

        dim3 gridSize((width + 16 - 1) / 16, (height + 16 - 1) / 16);
        dim3 blockSize(16, 16);
        d_bilateral_filter<<<gridSize, blockSize>>>(dDest, width, height, e_d,
                                                    radius, texObj);

        // sync host and stop computation timer
        checkCudaErrors(cudaDeviceSynchronize());
        dKernelTime += sdkGetTimerValue(&timer);

        if (iterations > 1) {
            // copy result back from global memory to array
            checkCudaErrors(cudaMemcpy2D(dTemp, pitch, dDest,
                                         sizeof(int) * width,
                                         sizeof(int) * width, height,
                                         cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaDestroyTextureObject(texObj));
            texObj = createTextureObject(dTemp, width, height);
        }
    }

    checkCudaErrors(cudaDestroyTextureObject(texObj));

    return ((dKernelTime / 1000.) / (double)iterations);
}
