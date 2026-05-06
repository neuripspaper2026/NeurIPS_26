/*-----------------------------------------------------------
 ** gaussian.cu -- The program is to solve a linear system Ax = b
 **   by using Gaussian Elimination. The algorithm on page 101
 **   ("Foundations of Parallel Programming") is used.
 **   The sequential version is gaussian.c.  This parallel
 **   implementation converts three independent for() loops
 **   into three Fans.  Use the data file ge_3.dat to verify
 **   the correction of the output.
 **
 ** Written by Andreas Kura, 02/15/95
 ** Modified by Chong-wei Xu, 04/20/95
 ** Modified by Chris Gregg for CUDA, 07/20/2009
 **-----------------------------------------------------------
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include "cuda.h"
#include <string.h>
#include <math.h>

#ifdef RD_WG_SIZE_0_0
#define MAXBLOCKSIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
#define MAXBLOCKSIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
#define MAXBLOCKSIZE RD_WG_SIZE
#else
#define MAXBLOCKSIZE 512
#endif

// 2D defines. Go from specific to general
#ifdef RD_WG_SIZE_1_0
#define BLOCK_SIZE_XY RD_WG_SIZE_1_0
#elif defined(RD_WG_SIZE_1)
#define BLOCK_SIZE_XY RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
#define BLOCK_SIZE_XY RD_WG_SIZE
#else
#define BLOCK_SIZE_XY 4
#endif

int Size;
float *a, *b, *finalVec;
float *m;

FILE *fp;

void InitProblemOnce(char *filename);
void InitPerRun();
void ForwardSub();
void BackSub();
__global__ void Fan1(float *m, float *a, int Size, int t);
__global__ void Fan2(float *m, float *a, float *b, int Size, int j1, int t);
void InitMat(float *ary, int nrow, int ncol);
void InitAry(float *ary, int ary_size);
void PrintMat(float *ary, int nrow, int ncolumn);
void PrintAry(float *ary, int ary_size);
void PrintDeviceProperties();
void checkCUDAError(const char *msg);

unsigned int totalKernelTime = 0;

// create both matrix and right hand side, Ke Wang 2013/08/12 11:51:06
void create_matrix(float *m, int size) {
    int i, j;
    float lamda = -0.01;
    float coe[2 * size - 1];
    float coe_i = 0.0;

    for (i = 0; i < size; i++) {
        coe_i = 10 * exp(lamda * i);
        j = size - 1 + i;
        coe[j] = coe_i;
        j = size - 1 - i;
        coe[j] = coe_i;
    }


    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            m[i * size + j] = coe[size - 1 - i + j];
        }
    }
}


int main(int argc, char *argv[]) {
    struct timespec main_start, main_end;
    clock_gettime(CLOCK_MONOTONIC, &main_start);
    
    printf("WG size of kernel 1 = %d, WG size of kernel 2= %d X %d\n",
           MAXBLOCKSIZE, BLOCK_SIZE_XY, BLOCK_SIZE_XY);
    int verbose = 1;
    int i, j;
    char flag;
    const char *output_file = NULL;
    
    if (argc < 2) {
        printf("Usage: gaussian -f filename / -s size [-q] [-o output_file]\n\n");
        printf(
            "-q (quiet) suppresses printing the matrix and result values.\n");
        printf("-f (filename) path of input file\n");
        printf("-s (size) size of matrix. Create matrix and rhs in this "
               "program \n");
        printf("-o (output) path of output file to save the solution\n");
        printf("The first line of the file contains the dimension of the "
               "matrix, n.");
        printf("The second line of the file is a newline.\n");
        printf(
            "The next n lines contain n tab separated values for the matrix.");
        printf("The next line of the file is a newline.\n");
        printf("The next line of the file is a 1xn vector with tab separated "
               "values.\n");
        printf("The next line of the file is a newline. (optional)\n");
        printf("The final line of the file is the pre-computed solution. "
               "(optional)\n");
        printf("Example: matrix4.txt:\n");
        printf("4\n");
        printf("\n");
        printf("-0.6	-0.5	0.7	0.3\n");
        printf("-0.3	-0.9	0.3	0.7\n");
        printf("-0.4	-0.5	-0.3	-0.8\n");
        printf("0.0	-0.1	0.2	0.9\n");
        printf("\n");
        printf("-0.85	-0.68	0.24	-0.53\n");
        printf("\n");
        printf("0.7	0.0	-0.4	-0.5\n");
        exit(0);
    }

    // PrintDeviceProperties();
    // char filename[100];
    // sprintf(filename,"matrices/matrix%d.txt",size);

    for (i = 1; i < argc; i++) {
        if (argv[i][0] == '-') { // flag
            flag = argv[i][1];
            switch (flag) {
            case 's': // platform
                i++;
                Size = atoi(argv[i]);
                printf("Create matrix internally in parse, size = %d \n", Size);

                a = (float *)malloc(Size * Size * sizeof(float));
                create_matrix(a, Size);

                b = (float *)malloc(Size * sizeof(float));
                for (j = 0; j < Size; j++)
                    b[j] = 1.0;

                m = (float *)malloc(Size * Size * sizeof(float));
                break;
            case 'f': // platform
                i++;
                printf("Read file from %s \n", argv[i]);
                InitProblemOnce(argv[i]);
                break;
            case 'q': // quiet
                verbose = 0;
                break;
            case 'o': // output file
                i++;
                output_file = argv[i];
                verbose = 0;  // Automatically enable quiet mode when output file is specified
                printf("Output will be saved to %s \n", output_file);
                break;
            }
        }
    }

    // InitProblemOnce(filename);
    InitPerRun();
    // begin timing
    struct timeval time_start;
    gettimeofday(&time_start, NULL);

    // run kernels
    ForwardSub();

    // end timing
    struct timeval time_end;
    gettimeofday(&time_end, NULL);
    unsigned int time_total =
        (time_end.tv_sec * 1000000 + time_end.tv_usec) -
        (time_start.tv_sec * 1000000 + time_start.tv_usec);

    if (verbose) {
        printf("Matrix m is: \n");
        PrintMat(m, Size, Size);

        printf("Matrix a is: \n");
        PrintMat(a, Size, Size);

        printf("Array b is: \n");
        PrintAry(b, Size);
    }
    BackSub();
    if (verbose) {
        printf("The final solution is: \n");
        PrintAry(finalVec, Size);
    }
    
    // Write output to file if specified
    if (output_file != NULL) {
        FILE *fp_out = fopen(output_file, "w");
        if (fp_out == NULL) {
            fprintf(stderr, "Error: Cannot open output file %s\n", output_file);
        } else {
            fprintf(fp_out, "Size: %d\n", Size);
            fprintf(fp_out, "Solution:\n");
            for (i = 0; i < Size; i++) {
                fprintf(fp_out, "x[%d] = %.6e\n", i, finalVec[i]);
            }
            fclose(fp_out);
            printf("Solution saved to %s\n", output_file);
        }
    }
    
    printf("\nTime total (including memory transfers)\t%f sec\n",
           time_total * 1e-6);
    printf("Time for CUDA kernels:\t%f sec\n", totalKernelTime * 1e-6);

    /*printf("%d,%d\n",size,time_total);
    fprintf(stderr,"%d,%d\n",size,time_total);*/

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
    
    fprintf(timing_file, "KERNEL_TIME: %.9f\n", totalKernelTime / 1e6);
    fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);
    
    if (timing_file != stderr)
        fclose(timing_file);

    free(m);
    free(a);
    free(b);
}
/*------------------------------------------------------
 ** PrintDeviceProperties
 **-----------------------------------------------------
 */
void PrintDeviceProperties() {
    cudaDeviceProp deviceProp;
    int nDevCount = 0;

    cudaGetDeviceCount(&nDevCount);
    printf("Total Device found: %d", nDevCount);
    for (int nDeviceIdx = 0; nDeviceIdx < nDevCount; ++nDeviceIdx) {
        memset(&deviceProp, 0, sizeof(deviceProp));
        if (cudaSuccess == cudaGetDeviceProperties(&deviceProp, nDeviceIdx)) {
            printf("\nDevice Name \t\t - %s ", deviceProp.name);
            printf("\n**************************************");
            printf("\nTotal Global Memory\t\t\t - %lu KB",
                   deviceProp.totalGlobalMem / 1024);
            printf("\nShared memory available per block \t - %lu KB",
                   deviceProp.sharedMemPerBlock / 1024);
            printf("\nNumber of registers per thread block \t - %d",
                   deviceProp.regsPerBlock);
            printf("\nWarp size in threads \t\t\t - %d", deviceProp.warpSize);
            printf("\nMemory Pitch \t\t\t\t - %zu bytes", deviceProp.memPitch);
            printf("\nMaximum threads per block \t\t - %d",
                   deviceProp.maxThreadsPerBlock);
            printf("\nMaximum Thread Dimension (block) \t - %d %d %d",
                   deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
                   deviceProp.maxThreadsDim[2]);
            printf("\nMaximum Thread Dimension (grid) \t - %d %d %d",
                   deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
                   deviceProp.maxGridSize[2]);
            printf("\nTotal constant memory \t\t\t - %zu bytes",
                   deviceProp.totalConstMem);
            printf("\nCUDA ver \t\t\t\t - %d.%d", deviceProp.major,
                   deviceProp.minor);
            printf("\nClock rate \t\t\t\t - %d KHz", deviceProp.clockRate);
            printf("\nTexture Alignment \t\t\t - %zu bytes",
                   deviceProp.textureAlignment);
            printf("\nDevice Overlap \t\t\t\t - %s",
                   deviceProp.deviceOverlap ? "Allowed" : "Not Allowed");
            printf("\nNumber of Multi processors \t\t - %d\n\n",
                   deviceProp.multiProcessorCount);
        } else
            printf("\n%s", cudaGetErrorString(cudaGetLastError()));
    }
}


/*------------------------------------------------------
 ** InitProblemOnce -- Initialize all of matrices and
 ** vectors by opening a data file specified by the user.
 **
 ** We used dynamic array *a, *b, and *m to allocate
 ** the memory storages.
 **------------------------------------------------------
 */
void InitProblemOnce(char *filename) {
    // char *filename = argv[1];

    // printf("Enter the data file name: ");
    // scanf("%s", filename);
    // printf("The file name is: %s\n", filename);

    fp = fopen(filename, "r");

    fscanf(fp, "%d", &Size);

    a = (float *)malloc(Size * Size * sizeof(float));

    InitMat(a, Size, Size);
    // printf("The input matrix a is:\n");
    // PrintMat(a, Size, Size);
    b = (float *)malloc(Size * sizeof(float));

    InitAry(b, Size);
    // printf("The input array b is:\n");
    // PrintAry(b, Size);

    m = (float *)malloc(Size * Size * sizeof(float));
}

/*------------------------------------------------------
 ** InitPerRun() -- Initialize the contents of the
 ** multipier matrix **m
 **------------------------------------------------------
 */
void InitPerRun() {
    int i;
    for (i = 0; i < Size * Size; i++)
        *(m + i) = 0.0;
}

/*-------------------------------------------------------
 ** Fan1() -- Calculate multiplier matrix
 ** Pay attention to the index.  Index i give the range
 ** which starts from 0 to range-1.  The real values of
 ** the index should be adjust and related with the value
 ** of t which is defined on the ForwardSub().
 **-------------------------------------------------------
 */
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void Fan1(float * __restrict__ m_cuda,
                     const float * __restrict__ a_cuda,
                     int Size, int t) {
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    int len  = Size - 1 - t;
    if (gidx >= len) return;

    int row   = gidx + t + 1;
    int baseR = row * Size;
    int diag  = t * Size + t;

    // Use local temporary for compiler to keep in registers
    float denom = a_cuda[diag];
    float num   = a_cuda[baseR + t];

    m_cuda[baseR + t] = num / denom;
}

/*-------------------------------------------------------
 ** Fan2() -- Modify the matrix A into LUD
 **-------------------------------------------------------
 */

__global__ void Fan2(const float * __restrict__ m_cuda,
                     float * __restrict__ a_cuda,
                     float * __restrict__ b_cuda,
                     int Size, int j1, int t) {
    int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    int yidx = blockIdx.y * blockDim.y + threadIdx.y;

    int lenx = Size - 1 - t;
    int leny = Size - t;

    if (xidx >= lenx || yidx >= leny) return;

    int row      = xidx + 1 + t;
    int col      = yidx + t;
    int rowBase  = row * Size;
    int tBase    = t * Size;

    float m_val  = m_cuda[rowBase + t];
    float a_tcol = a_cuda[tBase + col];

    // FMA-friendly form: a -= m * a_tcol
    float a_val  = a_cuda[rowBase + col];
    a_val -= m_val * a_tcol;
    a_cuda[rowBase + col] = a_val;

    if (yidx == 0) {
        // Only one thread per row updates b_cuda, avoid extra global reads
        float b_row = b_cuda[row];
        float b_t   = b_cuda[t];
        b_row -= m_val * b_t;
        b_cuda[row] = b_row;
    }
}

/*------------------------------------------------------
 ** ForwardSub() -- Forward substitution of Gaussian
 ** elimination.
 **------------------------------------------------------
 */
void ForwardSub() {
    int t;
    float *m_cuda, *a_cuda, *b_cuda;

    // allocate memory on GPU
    cudaMalloc((void **)&m_cuda, Size * Size * sizeof(float));

    cudaMalloc((void **)&a_cuda, Size * Size * sizeof(float));

    cudaMalloc((void **)&b_cuda, Size * sizeof(float));

    // copy memory to GPU
    cudaMemcpy(m_cuda, m, Size * Size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(a_cuda, a, Size * Size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_cuda, b, Size * sizeof(float), cudaMemcpyHostToDevice);

    int block_size, grid_size;

    block_size = MAXBLOCKSIZE;
    grid_size = (Size / block_size) + (!(Size % block_size) ? 0 : 1);
    // printf("1d grid size: %d\n",grid_size);


    dim3 dimBlock(block_size);
    dim3 dimGrid(grid_size);
    // dim3 dimGrid( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );

    int blockSize2d, gridSize2d;
    blockSize2d = BLOCK_SIZE_XY;
    gridSize2d = (Size / blockSize2d) + (!(Size % blockSize2d ? 0 : 1));

    dim3 dimBlockXY(blockSize2d, blockSize2d);
    dim3 dimGridXY(gridSize2d, gridSize2d);

    // begin timing kernels
    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);
    
    for (t = 0; t < (Size - 1); t++) {
        Fan1<<<dimGrid, dimBlock>>>(m_cuda, a_cuda, Size, t);
        Fan2<<<dimGridXY, dimBlockXY>>>(m_cuda, a_cuda, b_cuda, Size, Size - t,
                                        t);
        checkCUDAError("Fan2");
    }
    
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    double total_kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) + 
                                (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
    
    // Convert to microseconds for backward compatibility
    totalKernelTime = total_kernel_time * 1e6;

    // copy memory back to CPU
    cudaMemcpy(m, m_cuda, Size * Size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(a, a_cuda, Size * Size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b, b_cuda, Size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(m_cuda);
    cudaFree(a_cuda);
    cudaFree(b_cuda);
}

/*------------------------------------------------------
 ** BackSub() -- Backward substitution
 **------------------------------------------------------
 */

void BackSub() {
    // create a new vector to hold the final answer
    finalVec = (float *)malloc(Size * sizeof(float));
    // solve "bottom up"
    int i, j;
    for (i = 0; i < Size; i++) {
        finalVec[Size - i - 1] = b[Size - i - 1];
        for (j = 0; j < i; j++) {
            finalVec[Size - i - 1] -=
                *(a + Size * (Size - i - 1) + (Size - j - 1)) *
                finalVec[Size - j - 1];
        }
        finalVec[Size - i - 1] = finalVec[Size - i - 1] /
                                 *(a + Size * (Size - i - 1) + (Size - i - 1));
    }
}

void InitMat(float *ary, int nrow, int ncol) {
    int i, j;

    for (i = 0; i < nrow; i++) {
        for (j = 0; j < ncol; j++) {
            fscanf(fp, "%f", ary + Size * i + j);
        }
    }
}

/*------------------------------------------------------
 ** PrintMat() -- Print the contents of the matrix
 **------------------------------------------------------
 */
void PrintMat(float *ary, int nrow, int ncol) {
    int i, j;

    for (i = 0; i < nrow; i++) {
        for (j = 0; j < ncol; j++) {
            printf("%8.2f ", *(ary + Size * i + j));
        }
        printf("\n");
    }
    printf("\n");
}

/*------------------------------------------------------
 ** InitAry() -- Initialize the array (vector) by reading
 ** data from the data file
 **------------------------------------------------------
 */
void InitAry(float *ary, int ary_size) {
    int i;

    for (i = 0; i < ary_size; i++) {
        fscanf(fp, "%f", &ary[i]);
    }
}

/*------------------------------------------------------
 ** PrintAry() -- Print the contents of the array (vector)
 **------------------------------------------------------
 */
void PrintAry(float *ary, int ary_size) {
    int i;
    for (i = 0; i < ary_size; i++) {
        printf("%.2f ", ary[i]);
    }
    printf("\n\n");
}
void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
