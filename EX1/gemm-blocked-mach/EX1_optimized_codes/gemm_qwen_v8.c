#include <time.h>
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

    // Prefetch m1 and m2 data to improve cache performance
    __builtin_prefetch(m1, 0, 3);
    __builtin_prefetch(m2, 0, 3);

    // Loop optimizations: hoist invariants and minimize memory access
    for (jj = 0; jj < row_size; jj += block_size){
        for (kk = 0; kk < row_size; kk += block_size){
            for (i = 0; i < row_size; ++i){
                i_row = i * row_size;
                const TYPE* m1_ptr = &m1[i_row + kk];
                for (k = 0; k < block_size; ++k){
                    k_row = (k + kk) * row_size;
                    temp_x = m1_ptr[k];
                    TYPE* prod_ptr = &prod[i_row + jj];
                    const TYPE* m2_ptr = &m2[k_row + jj];
                    // Unroll innermost loop for better performance
                    j = 0;
                    for (; j < block_size - 3; j += 4) {
                        mul = temp_x * m2_ptr[j];
                        prod_ptr[j] += mul;
                        mul = temp_x * m2_ptr[j+1];
                        prod_ptr[j+1] += mul;
                        mul = temp_x * m2_ptr[j+2];
                        prod_ptr[j+2] += mul;
                        mul = temp_x * m2_ptr[j+3];
                        prod_ptr[j+3] += mul;
                    }
                    for (; j < block_size; ++j){
                        mul = temp_x * m2_ptr[j];
                        prod_ptr[j] += mul;
                    }
                }
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    gemm_blocked_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                    (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
