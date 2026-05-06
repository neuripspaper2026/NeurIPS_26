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
__global__ void d_bilateral_filter(uint *od, int w, int h, float e_d, int r,
                                   cudaTextureObject_t texObj) {
    // 2D thread indices
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Precompute the flat output index (kept even if out-of-bounds to avoid
    // control-flow divergence around the final store; invalid threads will
    // skip computation and store a zero)
    const int outIdx = y * w + x;

    // Check bounds once and keep a predicate to avoid early return
    const bool inBounds = (x < w) && (y < h);

    // Initialize accumulators
    float  sum = 0.0f;
    float4 t   = make_float4(0.f, 0.f, 0.f, 0.f);

    // Preload center pixel only for in-bounds threads
    float4 center = make_float4(0.f, 0.f, 0.f, 0.f);
    if (inBounds) {
        center = tex2D<float4>(texObj, static_cast<float>(x),
                               static_cast<float>(y));
    }

    // Unroll the inner loops to reduce loop overhead and help the compiler
    // generate efficient code on A100. Radius is assumed to be reasonably small
    // (<= 31 as limited by cGaussian size 64).
    const int rLocal = r;

    // Manually compute Gaussian row/col indices to reduce repeated address math
    #pragma unroll
    for (int i = -rLocal; i <= rLocal; ++i) {
        const int gi = i + rLocal;
        const float gRow = cGaussian[gi];

        const float yi = static_cast<float>(y + i);

        #pragma unroll
        for (int j = -rLocal; j <= rLocal; ++j) {
            if (!inBounds) {
                // Threads outside image bounds skip all work but still
                // participate in the store at the end (writing a neutral value)
                continue;
            }

            const int gj = j + rLocal;
            const float gCol = cGaussian[gj];

            // Texture fetch with FP coordinates (avoids integer->float
            // conversion inside tex unit and lets compiler hoist conversions)
            const float xj = static_cast<float>(x + j);
            float4 curPix  = tex2D<float4>(texObj, xj, yi);

            // Domain factor (Gaussian in x and y)
            float factor = gRow * gCol;

            // Range factor (Euclidean distance in color space)
            factor *= euclideanLen(curPix, center, e_d);

            // FMA-friendly accumulation
            t.x = fmaf(factor, curPix.x, t.x);
            t.y = fmaf(factor, curPix.y, t.y);
            t.z = fmaf(factor, curPix.z, t.z);
            t.w = fmaf(factor, curPix.w, t.w);

            sum += factor;
        }
    }

    // Normalize and store result; out-of-bounds threads write 0 (ignored by
    // caller because they are outside valid image region)
    if (inBounds && sum > 0.0f) {
        float invSum = __fdividef(1.0f, sum);
        float4 out   = make_float4(t.x * invSum,
                                   t.y * invSum,
                                   t.z * invSum,
                                   t.w * invSum);
        od[outIdx]   = rgbaFloatToInt(out);
    } else if (x < w && y < h) {
        // In rare case sum == 0, fall back to center pixel
        float4 out = center;
        od[outIdx] = rgbaFloatToInt(out);
    }
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
