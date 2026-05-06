#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../gemm.h"

static double gemm_blocked_kernel_time_acc = 0.0;

void reset_gemm_blocked_kernel_time(void) { gemm_blocked_kernel_time_acc = 0.0; }
double get_gemm_blocked_kernel_time(void) { return gemm_blocked_kernel_time_acc; }

void bbgemm(TYPE m1[N], TYPE m2[N], TYPE prod[N]){
    int i, k, j, jj, kk;
    int i_row, k_row;
    TYPE temp_x, mul;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    #pragma omp parallel for private(jj, kk, i, k, j, i_row, k_row, temp_x, mul) shared(m1, m2, prod) schedule(static)
    for (jj = 0; jj < row_size; jj += block_size){
        for (kk = 0; kk < row_size; kk += block_size){
            for (i = 0; i < row_size; ++i){
                for (k = 0; k < block_size; ++k){
                    i_row = i * row_size;
                    k_row = (k + kk) * row_size;
                    temp_x = m1[i_row + k + kk];
                    for (j = 0; j < block_size; ++j){
                        mul = temp_x * m2[k_row + j + jj];
                        #pragma omp atomic
                        prod[i_row + j + jj] += mul;
                    }
                }
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    gemm_blocked_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                    (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
