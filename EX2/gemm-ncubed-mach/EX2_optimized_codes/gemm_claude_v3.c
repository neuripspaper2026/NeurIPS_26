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

    #pragma omp parallel
    {
        // Thread-local transpose buffer for m2
        TYPE m2_T[N] __attribute__((aligned(64)));
        
        // Transpose m2 in parallel
        #pragma omp for collapse(2) schedule(static)
        for(i = 0; i < row_size; i++) {
            for(j = 0; j < col_size; j++) {
                m2_T[j * col_size + i] = m2[i * col_size + j];
            }
        }
        
        // Compute GEMM using transposed m2
        #pragma omp for schedule(static) nowait
        for(i = 0; i < row_size; i++) {
            int i_col = i * col_size;
            for(j = 0; j < col_size; j++) {
                int j_col = j * col_size;
                TYPE sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
                
                // Unroll inner loop by 4
                for(k = 0; k < row_size - 3; k += 4) {
                    sum0 += m1[i_col + k] * m2_T[j_col + k];
                    sum1 += m1[i_col + k + 1] * m2_T[j_col + k + 1];
                    sum2 += m1[i_col + k + 2] * m2_T[j_col + k + 2];
                    sum3 += m1[i_col + k + 3] * m2_T[j_col + k + 3];
                }
                
                // Handle remaining elements
                TYPE sum_remainder = 0;
                for(; k < row_size; k++) {
                    sum_remainder += m1[i_col + k] * m2_T[j_col + k];
                }
                
                prod[i_col + j] = sum0 + sum1 + sum2 + sum3 + sum_remainder;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    gemm_ncubed_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
