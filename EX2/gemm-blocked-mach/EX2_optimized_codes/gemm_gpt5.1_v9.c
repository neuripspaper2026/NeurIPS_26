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

    /* Precompute and restrict pointers to help the compiler optimize memory accesses */
    TYPE * __restrict m1p = m1;
    TYPE * __restrict m2p = m2;
    TYPE * __restrict prodp = prod;

    /* Parallelize outermost block loops; prod is updated in disjoint tiles */
#ifdef _OPENMP
#pragma omp parallel for collapse(2) private(jj, kk, i, k, j, i_row, k_row, temp_x, mul) \
    schedule(static)
#endif
    for (jj = 0; jj < row_size; jj += block_size){
        for (kk = 0; kk < row_size; kk += block_size){
            for (i = 0; i < row_size; ++i){
                i_row = i * row_size;
                for (k = 0; k < block_size; ++k){
                    k_row = (k + kk) * row_size;
                    temp_x = m1p[i_row + k + kk];
#pragma GCC ivdep
#pragma clang loop vectorize(enable)
                    for (j = 0; j < block_size; ++j){
                        mul = temp_x * m2p[k_row + j + jj];
                        prodp[i_row + j + jj] += mul;
                    }
                }
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    gemm_blocked_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                    (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
