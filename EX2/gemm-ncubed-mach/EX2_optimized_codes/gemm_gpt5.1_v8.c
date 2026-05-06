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

    /* Precompute row offsets to reduce integer multiplications */
    int row_offset[row_size];
    for (i = 0; i < row_size; i++) {
        row_offset[i] = i * col_size;
    }

    /* Use OpenMP for outer loops, collapse to increase workload per thread.
       Use private scalars and reduction on sum inside innermost loop. */
#ifdef _OPENMP
#pragma omp parallel for private(j, k) schedule(static)
#endif
    for(i = 0; i < row_size; i++) {
        int i_col = row_offset[i];
        for(j = 0; j < col_size; j++) {
            TYPE sum = 0.0;
            /* Unroll inner loop by 4 for better ILP and fewer loop overheads */
            int k_limit = row_size & ~3; /* largest multiple of 4 <= row_size */
            for(k = 0; k < k_limit; k += 4) {
                int k_col0 = k * col_size;
                int k_col1 = (k + 1) * col_size;
                int k_col2 = (k + 2) * col_size;
                int k_col3 = (k + 3) * col_size;

                sum += m1[i_col + k    ] * m2[k_col0 + j];
                sum += m1[i_col + k + 1] * m2[k_col1 + j];
                sum += m1[i_col + k + 2] * m2[k_col2 + j];
                sum += m1[i_col + k + 3] * m2[k_col3 + j];
            }
            /* Handle remaining iterations if row_size not multiple of 4 */
            for(; k < row_size; k++) {
                int k_col = k * col_size;
                sum += m1[i_col + k] * m2[k_col + j];
            }
            prod[i_col + j] = sum;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    gemm_ncubed_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
