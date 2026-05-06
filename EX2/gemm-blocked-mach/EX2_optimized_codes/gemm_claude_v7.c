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
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    #pragma omp parallel
    {
        #pragma omp for schedule(static) collapse(2) nowait
        loopjj:for (jj = 0; jj < row_size; jj += block_size){
            loopkk:for (kk = 0; kk < row_size; kk += block_size){
                loopi:for (i = 0; i < row_size; ++i){
                    int i_row = i * row_size;
                    TYPE temp[block_size];
                    
                    loopk:for (k = 0; k < block_size; ++k){
                        temp[k] = m1[i_row + k + kk];
                    }
                    
                    loopj:for (j = 0; j < block_size; ++j){
                        TYPE sum = 0.0;
                        loopk2:for (k = 0; k < block_size; ++k){
                            int k_row = (k + kk) * row_size;
                            sum += temp[k] * m2[k_row + j + jj];
                        }
                        prod[i_row + j + jj] += sum;
                    }
                }
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    gemm_blocked_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                    (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
