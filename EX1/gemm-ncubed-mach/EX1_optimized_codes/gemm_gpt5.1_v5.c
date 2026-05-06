#include <time.h>
#include "../gemm.h"

static double gemm_ncubed_kernel_time_acc = 0.0;

void reset_gemm_ncubed_kernel_time(void) { gemm_ncubed_kernel_time_acc = 0.0; }
double get_gemm_ncubed_kernel_time(void) { return gemm_ncubed_kernel_time_acc; }

void gemm( TYPE m1[N], TYPE m2[N], TYPE prod[N] ){
    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    for (int i = 0; i < row_size; ++i) {
        int i_row = i * col_size;
        for (int j = 0; j < col_size; ++j) {
            TYPE sum = 0;
            int j_offset = j;
            const TYPE *restrict m1_row = &m1[i_row];
            const TYPE *restrict m2_col = &m2[j_offset];
            for (int k = 0; k < row_size; ++k) {
                sum += m1_row[k] * m2_col[k * col_size];
            }
            prod[i_row + j] = sum;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    gemm_ncubed_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
