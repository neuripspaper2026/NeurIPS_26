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

/*
    Image bilateral filtering example

    This sample uses CUDA to perform a simple bilateral filter on an image.

    Bilateral filter is an edge-preserving nonlinear smoothing filter. There
    are three parameters distribute to the filter: gaussian delta, euclidean
    delta and iterations.

    When the euclidean delta increases, most of the fine texture will be
    filtered away, yet all contours are as crisp as in the original image.
    If the euclidean delta approximates to ∞, the filter becomes a normal
    gaussian filter. Fine texture will blur more with larger gaussian delta.
    Multiple iterations have the effect of flattening the colors in an
    image considerably, but without blurring edges, which produces a cartoon
    effect.

    To learn more details about this filter, please view C. Tomasi's "Bilateral
    Filtering for Gray and Color Images".

*/

#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_functions.h>
#include <time.h>

char *image_filename;

// Benchmark parameters
const int iCycles = 150;

// Algorithm parameters (can be overridden via CLI)
const int iterations = 1;
static float g_gaussian_delta = 4;
static float g_euclidean_delta = 0.1f;
static int g_filter_radius = 5;
static const char *g_output_file = NULL;

unsigned int width, height;
unsigned int *hImage = NULL;

StopWatchInterface *kernel_timer = NULL;

// Timing variables
double total_kernel_time = 0.0;

// Kernel API
extern "C" void initTexture(int width, int height, void *pImage);
extern "C" void freeTextures();
extern "C" double bilateralFilterRGBA(unsigned int *d_dest, int width,
                                      int height, float e_d, int radius,
                                      int iterations,
                                      StopWatchInterface *timer);
extern "C" void updateGaussian(float delta, int radius);

// BMP API
extern "C" void LoadBMPFile(uchar4 **dst, unsigned int *width,
                            unsigned int *height, const char *name);


////////////////////////////////////////////////////////////////////////////////
//! Run a simple benchmark test for CUDA
////////////////////////////////////////////////////////////////////////////////

int runBenchmark() {
    unsigned int *dResult;
    unsigned int *hResult =
        (unsigned int *)malloc(width * height * sizeof(unsigned int));
    size_t pitch;
    checkCudaErrors(cudaMallocPitch((void **)&dResult, &pitch,
                                    width * sizeof(unsigned int), height));
    sdkStartTimer(&kernel_timer);

    // warm-up
    bilateralFilterRGBA(dResult, width, height, g_euclidean_delta, g_filter_radius,
                        iterations, kernel_timer);
    checkCudaErrors(cudaDeviceSynchronize());

    // Start round-trip timer and process iCycles loops on the GPU
    double dProcessingTime = 0.0;
    printf("\nRunning BilateralFilterGPU for %d cycles...\n\n", iCycles);

    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    for (int i = 0; i < iCycles; i++) {
        dProcessingTime +=
            bilateralFilterRGBA(dResult, width, height, g_euclidean_delta,
                                g_filter_radius, iterations, kernel_timer);
    }

    checkCudaErrors(cudaDeviceSynchronize());
    
    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    total_kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) + 
                        (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;

    // check if kernel execution generated an error and sync host
    getLastCudaError("Error: bilateralFilterRGBA Kernel execution FAILED");
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&kernel_timer);

    // Get average computation time per cycle
    dProcessingTime /= (double)iCycles;

    // log testname, throughput, timing and config info
    printf("bilateralFilter-texture, Throughput = %.4f M RGBA Pixels/s, Time = "
           "%.5f s, Size = %u RGBA Pixels, NumDevsUsed = %u\n",
           (1.0e-6 * width * height) / dProcessingTime, dProcessingTime,
           (width * height), 1);
    printf("\n");

    // read back the results to system memory and write to file
    if (g_output_file != NULL) {
        cudaMemcpy2D(hResult, sizeof(unsigned int) * width, dResult, pitch,
                     sizeof(unsigned int) * width, height,
                     cudaMemcpyDeviceToHost);
        
        FILE *fp = fopen(g_output_file, "wb");
        if (fp == NULL) {
            fprintf(stderr, "Failed to open output file '%s'\n", g_output_file);
            exit(EXIT_FAILURE);
        }
        size_t written = fwrite(hResult, sizeof(unsigned int), width * height, fp);
        fclose(fp);
        printf("Saved %zu RGBA pixels to '%s'\n", written, g_output_file);
    } else if (getenv("OUTPUT")) {
        cudaMemcpy2D(hResult, sizeof(unsigned int) * width, dResult, pitch,
                     sizeof(unsigned int) * width, height,
                     cudaMemcpyDeviceToHost);
        sdkSavePPM4ub("output.ppm", (unsigned char *)hResult, width, height);
    }

    free(hResult);
    checkCudaErrors(cudaFree(dResult));

    return 0;
}

void loadImageData(const char *image_path) {
    LoadBMPFile((uchar4 **)&hImage, &width, &height, image_path);

    if (!hImage) {
        fprintf(stderr, "Error opening file '%s'\n", image_path);
        exit(EXIT_FAILURE);
    }

    printf("Loaded '%s', %d x %d pixels\n\n", image_path, width, height);
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

void initCuda() {
    updateGaussian(g_gaussian_delta, g_filter_radius);

    initTexture(width, height, hImage);
    sdkCreateTimer(&kernel_timer);
}

void cleanup() {
    sdkDeleteTimer(&kernel_timer);

    if (hImage)
        free(hImage);

    freeTextures();

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
}

int main(int argc, char **argv) {
    struct timespec main_start, main_end;
    clock_gettime(CLOCK_MONOTONIC, &main_start);
    
    printf("%s Starting...\n\n", argv[0]);

    // Usage: ./bilateralFilter <IMAGE> <e_d> <g_d> <radius> <output>
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <IMAGE> <euclidean_delta> <gaussian_delta> <filter_radius> <output_file>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    
    // Parse command-line arguments
    g_euclidean_delta = atof(argv[2]);
    g_gaussian_delta = atof(argv[3]);
    g_filter_radius = atoi(argv[4]);
    g_output_file = argv[5];
    
    printf("Parameters: euclidean_delta=%.2f, gaussian_delta=%.2f, filter_radius=%d\n",
           g_euclidean_delta, g_gaussian_delta, g_filter_radius);
    
    loadImageData(argv[1]);

    initCuda();

    int devID = findCudaDevice(argc, (const char **)argv);
    int error = runBenchmark();

    cleanup();

    // Timing output
    clock_gettime(CLOCK_MONOTONIC, &main_end);
    double main_time = (main_end.tv_sec - main_start.tv_sec) + 
                       (main_end.tv_nsec - main_start.tv_nsec) / 1e9;
    
    FILE *timing_file = stderr;
    const char *timing_path = getenv("TIMING_LOG_FILE");
    if (timing_path && timing_path[0] != '\0') {
        FILE *tmp = fopen(timing_path, "w");
        if (tmp)
            timing_file = tmp;
    }
    
    fprintf(timing_file, "KERNEL_TIME: %.9f\n", total_kernel_time);
    fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);
    
    if (timing_file != stderr)
        fclose(timing_file);

    exit(error == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}
