#include <time.h>
#include "../gemm.h"

static double gemm_ncubed_kernel_time_acc = 0.0;

void reset_gemm_ncubed_kernel_time(void) { gemm_ncubed_kernel_time_acc = 0.0; }
double get_gemm_ncubed_kernel_time(void) { return gemm_ncubed_kernel_time_acc; }

void gemm( TYPE m1[N], TYPE m2[N], TYPE prod[N] ){
    int i, j, k;
    int k_col, i_col;
    TYPE mult;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    TYPE temp_m2[col_size];  // Cache for m2 column

    outer:for(i=0;i<row_size;i++) {
        i_col = i * col_size;
        middle:for(j=0;j<col_size;j++) {
            TYPE sum = 0;
            // Preload m2 column values
            for(k=0; k<row_size; k++) {
                temp_m2[k] = m2[k * col_size + j];
            }
            inner:for(k=0;k<row_size;k++) {
                mult = m1[i_col + k] * temp_m2[k];
                sum += mult;
            }
            prod[i_col + j]  = sum;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    gemm_ncubed_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
