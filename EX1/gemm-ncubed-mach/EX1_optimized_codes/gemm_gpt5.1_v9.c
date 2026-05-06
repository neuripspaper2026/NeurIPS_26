#include <time.h>
#include "../gemm.h"

static double gemm_ncubed_kernel_time_acc = 0.0;

void reset_gemm_ncubed_kernel_time(void) { gemm_ncubed_kernel_time_acc = 0.0; }
double get_gemm_ncubed_kernel_time(void) { return gemm_ncubed_kernel_time_acc; }

void gemm( TYPE m1[N], TYPE m2[N], TYPE prod[N] ){
    int i, j, k;
    int i_col, k_col;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    for (i = 0; i < row_size; i++) {
        i_col = i * col_size;
        for (j = 0; j < col_size; j++) {
            prod[i_col + j] = 0.0;
        }
        for (k = 0; k < row_size; k++) {
            k_col = k * col_size;
            TYPE a_ik = m1[i_col + k];
            int base = k_col;
            int pbase = i_col;
            for (j = 0; j < col_size; j++) {
                prod[pbase + j] += a_ik * m2[base + j];
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    gemm_ncubed_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
