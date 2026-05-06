#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <matrix.h>
#include <clock.h>
#include <time.h>

// C (m x n) = A (m x p) * B (p x n)
#include <stddef.h>

void matmul(size_t m, size_t n, size_t p, double **A, double **B, double **C)
{
    // Initialization
    for (size_t i = 0; i < m; i++)
    {
        double *Ci = C[i];
        for (size_t j = 0; j < n; j++)
        {
            Ci[j] = 0.0;
        }
    }

    // Accumulation (i-k-j ordering, better locality and parallelizable)
    for (size_t i = 0; i < m; i++)
    {
        double *Ai = A[i];
        double *Ci = C[i];
        for (size_t k = 0; k < p; k++)
        {
            double aik = Ai[k];
            double *Bk = B[k];
            for (size_t j = 0; j < n; j++)
            {
                Ci[j] += aik * Bk[j];
            }
        }
    }
}

#ifdef _OPENMP
#include <omp.h>

void matmul_omp(size_t m, size_t n, size_t p, double **A, double **B, double **C)
{
    // Initialization (parallelized)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m; i++)
    {
        double *Ci = C[i];
        for (size_t j = 0; j < n; j++)
        {
            Ci[j] = 0.0;
        }
    }

    // Accumulation (parallelized over i, with cache-friendly i-k-j ordering)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m; i++)
    {
        double *Ai = A[i];
        double *Ci = C[i];
        for (size_t k = 0; k < p; k++)
        {
            double aik = Ai[k];
            double *Bk = B[k];
            for (size_t j = 0; j < n; j++)
            {
                Ci[j] += aik * Bk[j];
            }
        }
    }
}
#endif

int main(int argc, char *argv[])
{
    int param_iters = 1;

    if (argc < 2 || argc > 3)
    {
        printf("Usage: %s <n> [output_file]\n", argv[0]);
        printf("  <n> is the desired test size.\n");
        printf("  [output_file] (optional) is the output matrix file.\n");
        return 1;
    }

    // Reads the test parameters from the command line
    int param_n = 0;
    sscanf(argv[1], "%d", &param_n);
    
    // Output file name (default or user-specified)
    const char *output_file = (argc == 3) ? argv[2] : "matmul_gcc_output.txt";
    printf("- Input parameters\n");
    printf("n\t= %i\n", param_n);
    size_t rows = param_n, cols = param_n;

    /* 固定随机种子，保证可复现 */
    srand(12345);

    // Allocates input/output resources
    double **in1_mat = new_matrix(rows, cols);
    double **in2_mat = new_matrix(rows, cols);
    double **out_mat = new_matrix(rows, cols);
    if (!in1_mat || !in2_mat || !out_mat)
    {
        printf("Error: not enough memory to run the test using n = %i\n", param_n);
        return 1;
    }

    // Initializes data
    rand_matrix(in1_mat, rows, cols);
    rand_matrix(in2_mat, rows, cols);

    // Calls to the corresponding function to perform the computation
    printf("- Executing test...\n");
    struct timespec main_start, main_end;
    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &main_start);

    FILE *timing_file = stderr;
    const char *timing_path = getenv("TIMING_LOG_FILE");
    if (timing_path && timing_path[0] != '\0') {
        FILE *tmp = fopen(timing_path, "w");
        if (tmp)
            timing_file = tmp;
    }

    double time_start = getClock();
    // ================================================

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);
    for (int iters = 0; iters < param_iters; iters++)
    {
        matmul(rows, cols, cols, in1_mat, in2_mat, out_mat);
    }
    clock_gettime(CLOCK_MONOTONIC, &kernel_end);

    // ================================================
    double time_finish = getClock();

    // Prints an execution report
    double checksum = checksum_matrix(out_mat, rows, cols);
    printf("time (s)= %.6f\n", time_finish - time_start);
    printf("size\t= %i\n", param_n);
    printf("chksum\t= %.0f\n", checksum);
    if (param_iters > 1)
        printf("iters\t= %i\n", param_iters);

    FILE *fp = fopen(output_file, "w");
    if (fp != NULL)
    {
        for (size_t i = 0; i < rows; i++)
        {
            for (size_t j = 0; j < cols; j++)
            {
                fprintf(fp, "%.3f ", out_mat[i][j]);
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
        printf("- Output written to: %s\n", output_file);
    }
    else
    {
        fprintf(stderr, "Error: cannot write to %s\n", output_file);
    }

    // Release allocated resources
    delete_matrix(in1_mat);
    delete_matrix(in2_mat);
    delete_matrix(out_mat);

    clock_gettime(CLOCK_MONOTONIC, &main_end);
    double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                         (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
    double main_time = (main_end.tv_sec - main_start.tv_sec) +
                       (main_end.tv_nsec - main_start.tv_nsec) / 1e9;

    fprintf(timing_file, "KERNEL_TIME: %.9f\n", kernel_time);
    fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);

    if (timing_file != stderr)
        fclose(timing_file);

    return 0;
}
