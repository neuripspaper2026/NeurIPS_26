#include <time.h>
#include "../gemm.h"

static double gemm_ncubed_kernel_time_acc = 0.0;

void reset_gemm_ncubed_kernel_time(void) { gemm_ncubed_kernel_time_acc = 0.0; }
double get_gemm_ncubed_kernel_time(void) { return gemm_ncubed_kernel_time_acc; }

void gemm(TYPE m1[N], TYPE m2[N], TYPE prod[N]) {
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    /* Precompute and cache column pointers for m2 to improve locality */
    TYPE *m2_cols[col_size];
    for (int j = 0; j < col_size; ++j) {
        m2_cols[j] = &m2[j];
    }

    for (int i = 0; i < row_size; ++i) {
        int i_col = i * col_size;
        TYPE *row_m1 = &m1[i_col];
        TYPE *row_prod = &prod[i_col];

        for (int j = 0; j < col_size; ++j) {
            TYPE sum = 0.0;
            TYPE *col_m2 = m2_cols[j];

            /* Unroll k-loop by 4 for better ILP and fewer loop overheads */
            int k = 0;
            for (; k <= row_size - 4; k += 4) {
                sum += row_m1[k] * col_m2[k * col_size]
                     + row_m1[k + 1] * col_m2[(k + 1) * col_size]
                     + row_m1[k + 2] * col_m2[(k + 2) * col_size]
                     + row_m1[k + 3] * col_m2[(k + 3) * col_size];
            }
            for (; k < row_size; ++k) {
                sum += row_m1[k] * col_m2[k * col_size];
            }

            row_prod[j] = sum;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    gemm_ncubed_kernel_time_acc +=
        (kernel_end.tv_sec - kernel_start.tv_sec) +
        (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
