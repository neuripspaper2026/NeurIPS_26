#ifdef _OPENMP
#include <omp.h>
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <CRSMatrix.h>
#include <Matrix2D.h>
#include <Vector.h>

#ifdef _OPENMP
#include <omp.h>
#endif

double getClock();

// Compute sparse matrix-vector multiplication
void atmux(double *restrict val, double *restrict x, double *restrict y,
           int *restrict col_ind, int *restrict row_ptr, int n) {
    // Initialize y to 0
    #pragma omp parallel for if(n > 1024)
    for (int t = 0; t < n; t++) {
        y[t] = 0.0;
    }

    // y = A^T x
    // Parallelize over rows; use atomic to avoid write conflicts on y
    #pragma omp parallel for if(n > 32)
    for (int i = 0; i < n; i++) {
        const double xi = x[i];
        const int row_start = row_ptr[i];
        const int row_end   = row_ptr[i + 1];

        for (int k = row_start; k < row_end; k++) {
            const int col = col_ind[k];
            const double valk = val[k];
            const double contrib = xi * valk;
            #pragma omp atomic
            y[col] += contrib;
        }
    }
}

int main(int argc, char *argv[]) {
    double param_sparsity = 0.66;
    int param_iters = 10;

    if (argc < 2 || argc > 3) {
        printf("Usage: %s <n> [output_file]\n", argv[0]);
        printf("  <n> is the desired test size.\n");
        printf("  [output_file] (optional) is the output vector file.\n");
        return 0;
    }

    // Reads the test parameters from the command line
    unsigned long param_n = 0;
    sscanf(argv[1], "%lu", &param_n);
    const char *output_file = (argc == 3) ? argv[2] : NULL;
    // (silenced) input banner to keep only checksum in stdout

    // Fix random seed for reproducibility
    srand(12345);

    // Allocates input/output resources and initializes data (if needed)
    Vector *out_vec = Vector_new(param_n);
    Vector *in_vec = Vector_new(param_n);
    Vector_rand(in_vec);
    Matrix2D *denseMat = Matrix2D_new(param_n, param_n);
    Matrix2D_randSparse(denseMat, param_sparsity);
    CRSMatrix *in_sparseMat = CRSMatrix_from(denseMat);

    if (!in_vec || !out_vec || !denseMat || !in_sparseMat) {
        printf("Error: not enough memory to run the test using n = %lu\n", param_n);
        return 0;
    }

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

    // Calls the corresponding function to perform the computation
    // (silenced) execution banner
    double time_start = getClock();
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);
    // ================================================

    for (int iters = 0; iters < param_iters; iters++) {
        atmux(CRSMatrix_getData(in_sparseMat), Vector_getData(in_vec), Vector_getData(out_vec),
              CRSMatrix_colRef(in_sparseMat), CRSMatrix_rowRef(in_sparseMat), param_n);
    }
    clock_gettime(CLOCK_MONOTONIC, &kernel_end);

    // ================================================
    double time_finish = getClock();

    // Prints only checksum for correctness comparison
    double checksum = Vector_checksum(out_vec);
    printf("%.0f\n", checksum);

    // Write vector to file if output_file is specified
    if (output_file != NULL) {
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            double *data = Vector_getData(out_vec);
            long long size = Vector_getSize(out_vec);
            for (long long i = 0; i < size; i++) {
                fprintf(fp, "%.3f\n", data[i]);
            }
            fclose(fp);
            fprintf(stderr, "- Output written to: %s\n", output_file);
        } else {
            fprintf(stderr, "Error: cannot write to %s\n", output_file);
        }
    }

    // Release allocated resources
    Matrix2D_delete(denseMat);
    CRSMatrix_delete(in_sparseMat);
    Vector_delete(in_vec);
    Vector_delete(out_vec);

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

double getClock() {
#ifdef _OPENMP
    return omp_get_wtime();
#elif __linux__ || __APPLE__
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1.0e9;
#else
    // Warning: this clock is invalid for parallel applications
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}
