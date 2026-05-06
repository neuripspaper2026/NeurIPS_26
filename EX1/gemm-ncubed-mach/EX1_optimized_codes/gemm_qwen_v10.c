#include <time.h>
#include "../gemm.h"

static double gemm_ncubed_kernel_time_acc = 0.0;

void reset_gemm_ncubed_kernel_time(void) { gemm_ncubed_kernel_time_acc = 0.0; }
double get_gemm_ncubed_kernel_time(void) { return gemm_ncubed_kernel_time_acc; }

void gemm( TYPE m1[N], TYPE m2[N], TYPE prod[N] ){
    int i, j, k;
    TYPE *m1_ptr, *m2_ptr, *prod_ptr;
    TYPE sum, mult;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    m1_ptr = m1;
    m2_ptr = m2;
    prod_ptr = prod;

    for(i=0; i<row_size; i++) {
        for(j=0; j<col_size; j++) {
            sum = 0;
            TYPE *m1_row = m1_ptr + i * col_size;
            TYPE *m2_col_start = m2_ptr + j;
            
            for(k=0; k<row_size; k++) {
                mult = m1_row[k] * (*m2_col_start);
                sum += mult;
                m2_col_start += col_size;
            }
            prod_ptr[i * col_size + j] = sum;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    gemm_ncubed_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
