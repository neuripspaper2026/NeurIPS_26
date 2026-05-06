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

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for(i=0; i<row_size; i++) {
        for(j=0; j<col_size; j++) {
            int i_col = i * col_size;
            TYPE sum = 0;
            for(k=0; k<row_size; k++) {
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
