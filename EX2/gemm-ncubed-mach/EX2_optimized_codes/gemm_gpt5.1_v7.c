#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../gemm.h"

static double gemm_ncubed_kernel_time_acc = 0.0;

void reset_gemm_ncubed_kernel_time(void) { gemm_ncubed_kernel_time_acc = 0.0; }
double get_gemm_ncubed_kernel_time(void) { return gemm_ncubed_kernel_time_acc; }

void gemm(TYPE m1[N], TYPE m2[N], TYPE prod[N]) {
    int i, j, k;
    int k_col, i_col;
    TYPE mult;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    /* Precompute column-major access helper to avoid repeated mults */
    const int rsz = row_size;
    const int csz = col_size;

#ifdef _OPENMP
    /* Parallelize outer loop; each thread works on distinct rows of prod */
    #pragma omp parallel for private(i,j,k,k_col,i_col,mult) schedule(static)
#endif
    for (i = 0; i < rsz; i++) {
        i_col = i * csz;
        for (j = 0; j < csz; j++) {
            TYPE sum = 0;
            for (k = 0; k < rsz; k++) {
                k_col = k * csz;
                mult = m1[i_col + k] * m2[k_col + j];
                sum += mult;
            }
            prod[i_col + j] = sum;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    gemm_ncubed_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
