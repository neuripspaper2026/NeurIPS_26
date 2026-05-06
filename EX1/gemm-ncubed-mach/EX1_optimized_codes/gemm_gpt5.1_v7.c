#include <time.h>
#include "../gemm.h"

static double gemm_ncubed_kernel_time_acc = 0.0;

void reset_gemm_ncubed_kernel_time(void) { gemm_ncubed_kernel_time_acc = 0.0; }
double get_gemm_ncubed_kernel_time(void) { return gemm_ncubed_kernel_time_acc; }

void gemm( TYPE m1[N], TYPE m2[N], TYPE prod[N] ){
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    for (int i = 0; i < row_size; i++) {
        const int i_col = i * col_size;
        for (int j = 0; j < col_size; j++) {
            double sum0 = 0.0;
            double sum1 = 0.0;
            double sum2 = 0.0;
            double sum3 = 0.0;

            int k = 0;
            for (; k <= row_size - 4; k += 4) {
                const int k_col0 = k * col_size;
                const int k_col1 = (k + 1) * col_size;
                const int k_col2 = (k + 2) * col_size;
                const int k_col3 = (k + 3) * col_size;

                const double a0 = m1[i_col + k];
                const double a1 = m1[i_col + k + 1];
                const double a2 = m1[i_col + k + 2];
                const double a3 = m1[i_col + k + 3];

                sum0 += a0 * m2[k_col0 + j];
                sum1 += a1 * m2[k_col1 + j];
                sum2 += a2 * m2[k_col2 + j];
                sum3 += a3 * m2[k_col3 + j];
            }

            double sum = sum0 + sum1 + sum2 + sum3;

            for (; k < row_size; k++) {
                const int k_col = k * col_size;
                sum += m1[i_col + k] * m2[k_col + j];
            }

            prod[i_col + j] = sum;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    gemm_ncubed_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
