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

    /* OpenMP parallelization and serial optimizations:
       - collapse(2) to distribute i,j space
       - schedule(static) for predictable load balance
       - restrict pointers (via local qualified aliases) to help vectorization
       - move invariant computations outside inner loops
       - use local temporaries with minimal aliasing
    */
    TYPE * __restrict__ A = m1;
    TYPE * __restrict__ B = m2;
    TYPE * __restrict__ C = prod;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) private(i,j,k)
#endif
    for (i = 0; i < row_size; i++) {
        for (j = 0; j < col_size; j++) {
            int const i_col = i * col_size;
            TYPE sum = 0.0;
            /* hoist j to reduce address arithmetic in inner loop */
            int const j_idx = j;
            for (k = 0; k < row_size; k++) {
                int const k_col = k * col_size;
                sum += A[i_col + k] * B[k_col + j_idx];
            }
            C[i_col + j_idx] = sum;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    gemm_ncubed_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
