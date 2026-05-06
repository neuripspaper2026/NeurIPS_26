#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../gemm.h"

static double gemm_ncubed_kernel_time_acc = 0.0;

void reset_gemm_ncubed_kernel_time(void) { gemm_ncubed_kernel_time_acc = 0.0; }
double get_gemm_ncubed_kernel_time(void) { return gemm_ncubed_kernel_time_acc; }

void gemm( TYPE m1[N], TYPE m2[N], TYPE prod[N] ){
    int i, j, k;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    // Zero-initialize the output matrix to ensure correctness
    for (i = 0; i < row_size * col_size; ++i) {
        prod[i] = 0.0;
    }

    // Optimized GEMM: loop ordering and OpenMP parallelization
    // Access pattern:
    //   m1[i*col_size + k]  -> row-major, contiguous in k
    //   m2[k*col_size + j]  -> row-major, strided in k but contiguous in j
    //   prod[i*col_size + j]
    //
    // Use k-outer loop so that each (i,k) pair updates an entire row segment of prod
    // and allows efficient use of m1's row and m2's row.
    // Parallelize across i and k with a static schedule.
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) private(i,j,k) schedule(static)
#endif
    for (i = 0; i < row_size; ++i) {
        for (k = 0; k < row_size; ++k) {
            const TYPE a_val = m1[i * col_size + k];
            const int k_col = k * col_size;
            const int i_col = i * col_size;
            for (j = 0; j < col_size; ++j) {
                prod[i_col + j] += a_val * m2[k_col + j];
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    gemm_ncubed_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
