#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#include "../../../common_rodinia/cuda/profile_main.h"

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

#define STR_SIZE 256

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD (3.0e6)
/* required precision in degrees	*/
#define PRECISION 0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP 0.5

/* chip parameters	*/
float t_chip = 0.0005;
float chip_height = 0.016;
float chip_width = 0.016;
/* ambient temperature, assuming no package at all	*/
float amb_temp = 80.0;

/* define timer macros */
#define pin_stats_reset() startCycle()
#define pin_stats_pause(cycles) stopCycle(cycles)
#define pin_stats_dump(cycles) printf("timer: %Lu\n", cycles)


void fatal(const char* s) { fprintf(stderr, "error: %s\n", s); }

void writeoutput(float *vect, int grid_rows, int grid_cols, const std::string file) {
    int i, j, index = 0;
    FILE *fp;
    char str[STR_SIZE];

    if ((fp = fopen(file.c_str(), "w")) == 0)
        printf("The file was not opened\n");

    for (i = 0; i < grid_rows; i++)
        for (j = 0; j < grid_cols; j++) {

            sprintf(str, "%d\t%g\n", index, vect[i * grid_cols + j]);
            fputs(str, fp);
            index++;
        }

    fclose(fp);
}


void readinput(float *vect, int grid_rows, int grid_cols, char *file) {
    int i, j;
    FILE *fp;
    char str[STR_SIZE];
    float val;

    if ((fp = fopen(file, "r")) == 0)
        printf("The file was not opened\n");


    for (i = 0; i <= grid_rows - 1; i++)
        for (j = 0; j <= grid_cols - 1; j++) {
            fgets(str, STR_SIZE, fp);
            if (feof(fp))
                fatal("not enough lines in file");
            // if ((sscanf(str, "%d%f", &index, &val) != 2) || (index !=
            // ((i-1)*(grid_cols-2)+j-1)))
            if ((sscanf(str, "%f", &val) != 1))
                fatal("invalid file format");
            vect[i * grid_cols + j] = val;
        }

    fclose(fp);
}

#define IN_RANGE(x, min, max) ((x) >= (min) && (x) <= (max))
#define CLAMP_RANGE(x, min, max) x = (x < (min)) ? min : ((x > (max)) ? max : x)
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

__global__ void calculate_temp(int iteration,
                               float *power,
                               float *temp_src,
                               float *temp_dst,
                               int grid_cols,
                               int grid_rows,
                               int border_cols,
                               int border_rows,
                               float Cap,
                               float Rx, float Ry, float Rz, float step) {

    __shared__ float temp_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float power_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float temp_t[BLOCK_SIZE][BLOCK_SIZE];

    const float amb_temp = 80.0f;
    const float step_div_Cap = step / Cap;
    const float Rx_1 = 1.0f / Rx;
    const float Ry_1 = 1.0f / Ry;
    const float Rz_1 = 1.0f / Rz;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int small_block_rows = BLOCK_SIZE - (iteration << 1);
    const int small_block_cols = BLOCK_SIZE - (iteration << 1);

    const int blkY = small_block_rows * by - border_rows;
    const int blkX = small_block_cols * bx - border_cols;
    const int blkYmax = blkY + BLOCK_SIZE - 1;
    const int blkXmax = blkX + BLOCK_SIZE - 1;

    const int yidx = blkY + ty;
    const int xidx = blkX + tx;
    const int index = grid_cols * yidx + xidx;

    const bool in_bounds = (yidx >= 0) && (yidx < grid_rows) && 
                           (xidx >= 0) && (xidx < grid_cols);

    float temp_val = 0.0f;
    float power_val = 0.0f;

    if (in_bounds) {
        temp_val = temp_src[index];
        power_val = power[index];
    }
    temp_on_cuda[ty][tx] = temp_val;
    power_on_cuda[ty][tx] = power_val;
    
    __syncthreads();

    const int validYmin = (blkY < 0) ? -blkY : 0;
    const int validYmax = (blkYmax > grid_rows - 1) ? 
                          BLOCK_SIZE - 1 - (blkYmax - grid_rows + 1) : 
                          BLOCK_SIZE - 1;
    const int validXmin = (blkX < 0) ? -blkX : 0;
    const int validXmax = (blkXmax > grid_cols - 1) ? 
                          BLOCK_SIZE - 1 - (blkXmax - grid_cols + 1) : 
                          BLOCK_SIZE - 1;

    int N = ty - 1;
    int S = ty + 1;
    int W = tx - 1;
    int E = tx + 1;

    N = (N < validYmin) ? validYmin : N;
    S = (S > validYmax) ? validYmax : S;
    W = (W < validXmin) ? validXmin : W;
    E = (E > validXmax) ? validXmax : E;

    bool computed = false;
    
    #pragma unroll 4
    for (int i = 0; i < iteration; i++) {
        const int i_plus_1 = i + 1;
        const int block_size_minus_i_2 = BLOCK_SIZE - i - 2;
        
        const bool tx_in_range = (tx >= i_plus_1) && (tx <= block_size_minus_i_2);
        const bool ty_in_range = (ty >= i_plus_1) && (ty <= block_size_minus_i_2);
        const bool valid_x_range = (tx >= validXmin) && (tx <= validXmax);
        const bool valid_y_range = (ty >= validYmin) && (ty <= validYmax);
        
        computed = tx_in_range && ty_in_range && valid_x_range && valid_y_range;
        
        if (computed) {
            const float temp_center = temp_on_cuda[ty][tx];
            const float temp_n = temp_on_cuda[N][tx];
            const float temp_s = temp_on_cuda[S][tx];
            const float temp_w = temp_on_cuda[ty][W];
            const float temp_e = temp_on_cuda[ty][E];
            
            const float laplacian_y = (temp_s + temp_n - 2.0f * temp_center) * Ry_1;
            const float laplacian_x = (temp_e + temp_w - 2.0f * temp_center) * Rx_1;
            const float ambient_term = (amb_temp - temp_center) * Rz_1;
            
            temp_t[ty][tx] = temp_center + 
                             step_div_Cap * (power_on_cuda[ty][tx] + 
                                            laplacian_y + 
                                            laplacian_x + 
                                            ambient_term);
        }
        
        __syncthreads();
        
        if (i < iteration - 1) {
            if (computed) {
                temp_on_cuda[ty][tx] = temp_t[ty][tx];
            }
            __syncthreads();
        }
    }

    if (computed) {
        temp_dst[index] = temp_t[ty][tx];
    }
}

/*
   compute N time steps
*/

int compute_tran_temp(float *MatrixPower, float *MatrixTemp[2], int col,
                      int row, int total_iterations, int num_iterations,
                      int blockCols, int blockRows, int borderCols,
                      int borderRows) {
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(blockCols, blockRows);

    float grid_height = chip_height / row;
    float grid_width = chip_width / col;

    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
    float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
    float Rz = t_chip / (K_SI * grid_height * grid_width);

    float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    float step = PRECISION / max_slope;
    float t;

    int src = 1, dst = 0;

    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    for (t = 0; t < total_iterations; t += num_iterations) {
        int temp = src;
        src = dst;
        dst = temp;
        PROFILE((
            calculate_temp<<<dimGrid, dimBlock>>>(
                MIN(num_iterations, total_iterations - t), MatrixPower,
                MatrixTemp[src], MatrixTemp[dst], col, row, borderCols, borderRows,
                Cap, Rx, Ry, Rz, step)
        ));
    }
    
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    g_kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) + 
                    (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
    
    return dst;
}

void usage(int argc, char **argv) {
    fprintf(stderr, "Usage: %s <grid_rows> <grid_cols> <pyramid_height> "
                    "<sim_time> <temp_file> <power_file> <output_file>\n",
            argv[0]);
    fprintf(stderr, "\t<grid_rows>  - number of rows in the "
                    "grid (positive integer)\n");
    fprintf(stderr, "\t<grid_cols>  - number of cols in the "
                    "grid (positive integer)\n");
    fprintf(stderr, "\t<pyramid_height> - pyramid heigh(positive integer)\n");
    fprintf(stderr, "\t<sim_time>   - number of iterations\n");
    fprintf(stderr, "\t<temp_file>  - name of the file containing the initial "
                    "temperature values of each cell\n");
    fprintf(stderr, "\t<power_file> - name of the file containing the "
                    "dissipated power values of each cell\n");
    fprintf(stderr, "\t<output_file> - name of the output file\n");
    exit(1);
}

void run(int argc, char **argv) {
    printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);

    int size;
    int grid_rows, grid_cols;
    float *FilesavingTemp, *FilesavingPower, *MatrixOut;
    char *tfile, *pfile, *ofile;

    int total_iterations = 60;
    int pyramid_height = 1; // number of iterations

    if (argc != 8)
        usage(argc, argv);
    if ((grid_rows = atoi(argv[1])) <= 0 || (grid_cols = atoi(argv[2])) <= 0 ||
        (pyramid_height = atoi(argv[3])) <= 0 ||
        (total_iterations = atoi(argv[4])) <= 0)
        usage(argc, argv);

    tfile = argv[5];
    pfile = argv[6];
    ofile = argv[7];

    size = grid_rows * grid_cols;

    /* --------------- pyramid parameters --------------- */
    #define EXPAND_RATE 2 // add one iteration will extend the pyramid base by 2 per each borderline
    int borderCols = (pyramid_height)*EXPAND_RATE / 2;
    int borderRows = (pyramid_height)*EXPAND_RATE / 2;
    int smallBlockCol = BLOCK_SIZE - (pyramid_height)*EXPAND_RATE;
    int smallBlockRow = BLOCK_SIZE - (pyramid_height)*EXPAND_RATE;
    int blockCols =
        grid_cols / smallBlockCol + ((grid_cols % smallBlockCol == 0) ? 0 : 1);
    int blockRows =
        grid_rows / smallBlockRow + ((grid_rows % smallBlockRow == 0) ? 0 : 1);

    FilesavingTemp = (float *)malloc(size * sizeof(float));
    FilesavingPower = (float *)malloc(size * sizeof(float));
    MatrixOut = (float *)calloc(size, sizeof(float));

    if (!FilesavingPower || !FilesavingTemp || !MatrixOut)
        fatal("unable to allocate memory");

    printf("pyramidHeight: %d\ngridSize: [%d, %d]\nborder:[%d, "
           "%d]\nblockGrid:[%d, %d]\ntargetBlock:[%d, %d]\n",
           pyramid_height, grid_cols, grid_rows, borderCols, borderRows,
           blockCols, blockRows, smallBlockCol, smallBlockRow);

    readinput(FilesavingTemp, grid_rows, grid_cols, tfile);
    readinput(FilesavingPower, grid_rows, grid_cols, pfile);

    float *MatrixTemp[2], *MatrixPower;
    cudaMalloc((void **)&MatrixTemp[0], sizeof(float) * size);
    cudaMalloc((void **)&MatrixTemp[1], sizeof(float) * size);
    cudaMemcpy(MatrixTemp[0], FilesavingTemp, sizeof(float) * size,
               cudaMemcpyHostToDevice);

    cudaMalloc((void **)&MatrixPower, sizeof(float) * size);
    cudaMemcpy(MatrixPower, FilesavingPower, sizeof(float) * size,
               cudaMemcpyHostToDevice);
    printf("Start computing the transient temperature\n");
    int ret = compute_tran_temp(MatrixPower, MatrixTemp, grid_cols, grid_rows,
                                total_iterations, pyramid_height, blockCols,
                                blockRows, borderCols, borderRows);
    printf("Ending simulation\n");
    cudaMemcpy(MatrixOut, MatrixTemp[ret], sizeof(float) * size,
               cudaMemcpyDeviceToHost);

    writeoutput(MatrixOut, grid_rows, grid_cols, ofile);

    cudaFree(MatrixPower);
    cudaFree(MatrixTemp[0]);
    cudaFree(MatrixTemp[1]);
    free(MatrixOut);
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
    
    fprintf(timing_file, "KERNEL_TIME: %.9f\n", g_kernel_time);
    fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);
    
    if (timing_file != stderr)
        fclose(timing_file);

    return EXIT_SUCCESS;
}
