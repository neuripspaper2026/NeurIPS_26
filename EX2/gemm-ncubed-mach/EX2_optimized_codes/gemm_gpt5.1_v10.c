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

    /* Precompute row-major base index for each row of m1/prod and m2 */
    int row_base[row_size];
    int col_base[row_size];
    for (i = 0; i < row_size; ++i) {
        row_base[i] = i * col_size;
        col_base[i] = i * col_size;
    }

    /* Zero-initialize product matrix to enable sum accumulation with fma */
    for (i = 0; i < row_size; ++i) {
        int ib = row_base[i];
        for (j = 0; j < col_size; ++j) {
            prod[ib + j] = 0;
        }
    }

    /* Optimized GEMM: i-k-j ordering for better cache reuse and OpenMP parallelism */
#ifdef _OPENMP
#pragma omp parallel for private(i,j,k) schedule(static)
#endif
    for (i = 0; i < row_size; ++i) {
        int ib = row_base[i];
        for (k = 0; k < row_size; ++k) {
            int kb = col_base[k];
            TYPE a_ik = m1[ib + k];
            /* Manual unrolling over j for better vectorization */
            int j_end = col_size & ~3;
            for (j = 0; j < j_end; j += 4) {
                TYPE b0 = m2[kb + j];
                TYPE b1 = m2[kb + j + 1];
                TYPE b2 = m2[kb + j + 2];
                TYPE b3 = m2[kb + j + 3];

                prod[ib + j]     += a_ik * b0;
                prod[ib + j + 1] += a_ik * b1;
                prod[ib + j + 2] += a_ik * b2;
                prod[ib + j + 3] += a_ik * b3;
            }
            for (; j < col_size; ++j) {
                prod[ib + j] += a_ik * m2[kb + j];
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    gemm_ncubed_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
