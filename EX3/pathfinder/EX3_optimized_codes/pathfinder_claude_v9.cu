#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <string.h>

#include "../../common_rodinia/cuda/profile_main.h"

// Global variable for kernel timing
double g_kernel_time = 0.0;

#define BLOCK_SIZE 256
#define HALO 1  // halo width along one direction when advancing to the next iteration

int rows, cols;
int *data;
int **wall;
int *result;
int pyramid_height;
const char *output_file = NULL;

void init(int argc, char **argv) {
    // 解析参数
    if (argc >= 4) {
        cols = atoi(argv[1]);
        rows = atoi(argv[2]);
        pyramid_height = atoi(argv[3]);
        
        // 查找 -o 参数
        for (int i = 4; i < argc - 1; i++) {
            if (strcmp(argv[i], "-o") == 0) {
                output_file = argv[i + 1];
                break;
            }
        }
    } else {
        printf("Usage: dynproc row_len col_len pyramid_height [--no-rand] [-o <output_file>]\n");
        exit(1);
    }
    data = new int[rows * cols];
    wall = new int *[rows];
    for (int n = 0; n < rows; n++)
        wall[n] = data + cols * n;
    result = new int[cols];

    srand(7);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            wall[i][j] = rand() % 10;
        }
    }

    if (output_file) {
        FILE *file = fopen(output_file, "w");
        if (!file) {
            fprintf(stderr, "Error: Cannot open output file %s\n", output_file);
            exit(1);
        }

        fprintf(file, "wall:\n");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                fprintf(file, "%d ", wall[i][j]);
            }
            fprintf(file, "\n");
        }

        fclose(file);
    }
}

#define IN_RANGE(x, min, max) ((x) >= (min) && (x) <= (max))
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <string.h>

#include "../../common_rodinia/cuda/profile_main.h"

// Global variable for kernel timing
double g_kernel_time = 0.0;

#define BLOCK_SIZE 256
#define HALO 1  // halo width along one direction when advancing to the next iteration

int rows, cols;
int *data;
int **wall;
int *result;
int pyramid_height;
const char *output_file = NULL;

void init(int argc, char **argv) {
    // 解析参数
    if (argc >= 4) {
        cols = atoi(argv[1]);
        rows = atoi(argv[2]);
        pyramid_height = atoi(argv[3]);
        
        // 查找 -o 参数
        for (int i = 4; i < argc - 1; i++) {
            if (strcmp(argv[i], "-o") == 0) {
                output_file = argv[i + 1];
                break;
            }
        }
    } else {
        printf("Usage: dynproc row_len col_len pyramid_height [--no-rand] [-o <output_file>]\n");
        exit(1);
    }
    data = new int[rows * cols];
    wall = new int *[rows];
    for (int n = 0; n < rows; n++)
        wall[n] = data + cols * n;
    result = new int[cols];

    srand(7);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            wall[i][j] = rand() % 10;
        }
    }

    if (output_file) {
        FILE *file = fopen(output_file, "w");
        if (!file) {
            fprintf(stderr, "Error: Cannot open output file %s\n", output_file);
            exit(1);
        }

        fprintf(file, "wall:\n");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                fprintf(file, "%d ", wall[i][j]);
            }
            fprintf(file, "\n");
        }

        fclose(file);
    }
}

#define IN_RANGE(x, min, max) ((x) >= (min) && (x) <= (max))
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

__global__ void dynproc_kernel(int iteration, int *gpuWall, int *gpuSrc,
                               int *gpuResults, int cols, int rows,
                               int startStep, int border) {

    __shared__ int prev[BLOCK_SIZE];
    __shared__ int result[BLOCK_SIZE];

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    const int warpId = tx >> 5;
    const int laneId = tx & 31;

    // calculate the small block size
    int small_block_cols = BLOCK_SIZE - iteration * HALO * 2;

    // calculate the boundary for the block according to
    // the boundary of its small block
    int blkX = small_block_cols * bx - border;
    int blkXmax = blkX + BLOCK_SIZE - 1;

    // calculate the global thread coordination
    int xidx = blkX + tx;

    // effective range within this block that falls within
    // the valid range of the input data
    // used to rule out computation outside the boundary.
    int validXmin = (blkX < 0) ? -blkX : 0;
    int validXmax = (blkXmax > cols - 1) ? BLOCK_SIZE - 1 - (blkXmax - cols + 1)
                                         : BLOCK_SIZE - 1;

    int W = tx - 1;
    int E = tx + 1;

    W = (W < validXmin) ? validXmin : W;
    E = (E > validXmax) ? validXmax : E;

    bool isValid = IN_RANGE(tx, validXmin, validXmax);

    // Coalesced global memory access
    if (IN_RANGE(xidx, 0, cols - 1)) {
        prev[tx] = gpuSrc[xidx];
    }

    __syncthreads();

    bool computed = false;
    int local_result = 0;
    
    #pragma unroll 4
    for (int i = 0; i < iteration; i++) {
        bool should_compute = IN_RANGE(tx, i + 1, BLOCK_SIZE - i - 2) && isValid;
        
        if (should_compute) {
            computed = true;
            int left = prev[W];
            int up = prev[tx];
            int right = prev[E];
            int shortest = min(min(left, up), right);
            int index = cols * (startStep + i) + xidx;
            local_result = shortest + gpuWall[index];
            result[tx] = local_result;
        }
        
        __syncthreads();
        
        if (i < iteration - 1) {
            if (should_compute) {
                prev[tx] = local_result;
            }
            __syncthreads();
        }
    }

    // Coalesced global memory write
    if (computed) {
        gpuResults[xidx] = local_result;
    }
}

/*
   compute N time steps
*/
int calc_path(int *gpuWall, int *gpuResult[2], int rows, int cols,
              int pyramid_height, int blockCols, int borderCols) {
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(blockCols);

    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    int src = 1, dst = 0;
    for (int t = 0; t < rows - 1; t += pyramid_height) {
        int temp = src;
        src = dst;
        dst = temp;
        PROFILE((
            dynproc_kernel<<<dimGrid, dimBlock>>>(
                MIN(pyramid_height, rows - t - 1), gpuWall, gpuResult[src],
                gpuResult[dst], cols, rows, t, borderCols)
        ));
    }
    
    cudaDeviceSynchronize();
    
    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) + 
                         (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
    g_kernel_time += kernel_time;

    return dst;
}

int run(int argc, char **argv) {
    init(argc, argv);

    /* --------------- pyramid parameters --------------- */
    int borderCols = (pyramid_height)*HALO;
    int smallBlockCol = BLOCK_SIZE - (pyramid_height)*HALO * 2;
    int blockCols =
        cols / smallBlockCol + ((cols % smallBlockCol == 0) ? 0 : 1);

    printf("pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: "
           "%d\nblockGrid:[%d]\ntargetBlock:[%d]\n",
           pyramid_height, cols, borderCols, BLOCK_SIZE, blockCols,
           smallBlockCol);

    int *gpuWall, *gpuResult[2];
    int size = rows * cols;

    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);

    cudaMalloc((void **)&gpuResult[0], sizeof(int) * cols);
    cudaMalloc((void **)&gpuResult[1], sizeof(int) * cols);
    cudaMemcpy(gpuResult[0], data, sizeof(int) * cols, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&gpuWall, sizeof(int) * (size - cols));
    cudaMemcpy(gpuWall, data + cols, sizeof(int) * (size - cols),
               cudaMemcpyHostToDevice);


    int final_ret = calc_path(gpuWall, gpuResult, rows, cols, pyramid_height,
                              blockCols, borderCols);

    cudaMemcpy(result, gpuResult[final_ret], sizeof(int) * cols,
               cudaMemcpyDeviceToHost);

    clock_gettime(CLOCK_REALTIME, &end);
    double elapsed = (end.tv_sec - start.tv_sec)
        + (end.tv_nsec - start.tv_nsec)/1E9;
    printf("%.6f seconds\n", elapsed);

    if (output_file) {
        FILE *file = fopen(output_file, "a");
        if (!file) {
            fprintf(stderr, "Error: Cannot open output file %s\n", output_file);
            exit(1);
        }

        fprintf(file, "data:\n");
        for (int i = 0; i < cols; i++)
            fprintf(file, "%d ", data[i]);
        fprintf(file, "\n");

        fprintf(file, "result:\n");
        for (int i = 0; i < cols; i++)
            fprintf(file, "%d ", result[i]);
        fprintf(file, "\n");

        fclose(file);
    }

    cudaFree(gpuWall);
    cudaFree(gpuResult[0]);
    cudaFree(gpuResult[1]);

    delete[] data;
    delete[] wall;
    delete[] result;

    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    struct timespec main_start, main_end;
    clock_gettime(CLOCK_MONOTONIC, &main_start);

    run(argc, argv);

    if (getenv("PROFILE")) {
        // warm up
        for (int i = 0; i < 5; i++)
            run(argc, argv);

        checkCudaErrors(cudaProfilerStart());
        nvtxRangePushA("host");

        run(argc, argv);

        nvtxRangePop();
        checkCudaErrors(cudaProfilerStop());
    }

    clock_gettime(CLOCK_MONOTONIC, &main_end);
    double main_time = (main_end.tv_sec - main_start.tv_sec) + 
                       (main_end.tv_nsec - main_start.tv_nsec) / 1e9;
    
    extern double g_kernel_time;
    
    FILE *timing_file = stderr;
    const char *timing_path = getenv("TIMING_LOG_FILE");
    if (timing_path && timing_path[0] != '\0') {
        FILE *tmp = fopen(timing_path, "w");
        if (tmp)
            timing_file = tmp;
    }
    
    fprintf(timing_file, "KERNEL_TIME: %.9f\n", g_kernel_time);
    fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);
    
    if (timing_file != stderr)
        fclose(timing_file);

    return EXIT_SUCCESS;
}

/*
   compute N time steps
*/
int calc_path(int *gpuWall, int *gpuResult[2], int rows, int cols,
              int pyramid_height, int blockCols, int borderCols) {
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(blockCols);

    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    int src = 1, dst = 0;
    for (int t = 0; t < rows - 1; t += pyramid_height) {
        int temp = src;
        src = dst;
        dst = temp;
        PROFILE((
            dynproc_kernel<<<dimGrid, dimBlock>>>(
                MIN(pyramid_height, rows - t - 1), gpuWall, gpuResult[src],
                gpuResult[dst], cols, rows, t, borderCols)
        ));
    }
    
    cudaDeviceSynchronize();
    
    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) + 
                         (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
    g_kernel_time += kernel_time;

    return dst;
}

int run(int argc, char **argv) {
    init(argc, argv);

    /* --------------- pyramid parameters --------------- */
    int borderCols = (pyramid_height)*HALO;
    int smallBlockCol = BLOCK_SIZE - (pyramid_height)*HALO * 2;
    int blockCols =
        cols / smallBlockCol + ((cols % smallBlockCol == 0) ? 0 : 1);

    printf("pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: "
           "%d\nblockGrid:[%d]\ntargetBlock:[%d]\n",
           pyramid_height, cols, borderCols, BLOCK_SIZE, blockCols,
           smallBlockCol);

    int *gpuWall, *gpuResult[2];
    int size = rows * cols;

    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);

    cudaMalloc((void **)&gpuResult[0], sizeof(int) * cols);
    cudaMalloc((void **)&gpuResult[1], sizeof(int) * cols);
    cudaMemcpy(gpuResult[0], data, sizeof(int) * cols, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&gpuWall, sizeof(int) * (size - cols));
    cudaMemcpy(gpuWall, data + cols, sizeof(int) * (size - cols),
               cudaMemcpyHostToDevice);


    int final_ret = calc_path(gpuWall, gpuResult, rows, cols, pyramid_height,
                              blockCols, borderCols);

    cudaMemcpy(result, gpuResult[final_ret], sizeof(int) * cols,
               cudaMemcpyDeviceToHost);

    clock_gettime(CLOCK_REALTIME, &end);
    double elapsed = (end.tv_sec - start.tv_sec)
        + (end.tv_nsec - start.tv_nsec)/1E9;
    printf("%.6f seconds\n", elapsed);

    if (output_file) {
        FILE *file = fopen(output_file, "a");
        if (!file) {
            fprintf(stderr, "Error: Cannot open output file %s\n", output_file);
            exit(1);
        }

        fprintf(file, "data:\n");
        for (int i = 0; i < cols; i++)
            fprintf(file, "%d ", data[i]);
        fprintf(file, "\n");

        fprintf(file, "result:\n");
        for (int i = 0; i < cols; i++)
            fprintf(file, "%d ", result[i]);
        fprintf(file, "\n");

        fclose(file);
    }

    cudaFree(gpuWall);
    cudaFree(gpuResult[0]);
    cudaFree(gpuResult[1]);

    delete[] data;
    delete[] wall;
    delete[] result;

    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    struct timespec main_start, main_end;
    clock_gettime(CLOCK_MONOTONIC, &main_start);

    run(argc, argv);

    if (getenv("PROFILE")) {
        // warm up
        for (int i = 0; i < 5; i++)
            run(argc, argv);

        checkCudaErrors(cudaProfilerStart());
        nvtxRangePushA("host");

        run(argc, argv);

        nvtxRangePop();
        checkCudaErrors(cudaProfilerStop());
    }

    clock_gettime(CLOCK_MONOTONIC, &main_end);
    double main_time = (main_end.tv_sec - main_start.tv_sec) + 
                       (main_end.tv_nsec - main_start.tv_nsec) / 1e9;
    
    extern double g_kernel_time;
    
    FILE *timing_file = stderr;
    const char *timing_path = getenv("TIMING_LOG_FILE");
    if (timing_path && timing_path[0] != '\0') {
        FILE *tmp = fopen(timing_path, "w");
        if (tmp)
            timing_file = tmp;
    }
    
    fprintf(timing_file, "KERNEL_TIME: %.9f\n", g_kernel_time);
    fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);
    
    if (timing_file != stderr)
        fclose(timing_file);

    return EXIT_SUCCESS;
}
