#include <time.h>
#include "../gemm.h"

static double gemm_blocked_kernel_time_acc = 0.0;

void reset_gemm_blocked_kernel_time(void) { gemm_blocked_kernel_time_acc = 0.0; }
double get_gemm_blocked_kernel_time(void) { return gemm_blocked_kernel_time_acc; }

void bbgemm(TYPE m1[N], TYPE m2[N], TYPE prod[N]){
    int i, k, j, jj, kk;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    for (jj = 0; jj < row_size; jj += block_size){
        for (kk = 0; kk < row_size; kk += block_size){
            for (i = 0; i < row_size; ++i){
                const int i_row = i * row_size;
                for (k = 0; k < block_size; ++k){
                    const int k_index = k + kk;
                    const int k_row = k_index * row_size;
                    const TYPE temp_x = m1[i_row + k_index];
                    const int base_prod = i_row + jj;
                    const int base_m2   = k_row + jj;

                    /* Manually unroll j-loop for block_size == 8 */
                    TYPE mul0 = temp_x * m2[base_m2 + 0];
                    prod[base_prod + 0] += mul0;

                    TYPE mul1 = temp_x * m2[base_m2 + 1];
                    prod[base_prod + 1] += mul1;

                    TYPE mul2 = temp_x * m2[base_m2 + 2];
                    prod[base_prod + 2] += mul2;

                    TYPE mul3 = temp_x * m2[base_m2 + 3];
                    prod[base_prod + 3] += mul3;

                    TYPE mul4 = temp_x * m2[base_m2 + 4];
                    prod[base_prod + 4] += mul4;

                    TYPE mul5 = temp_x * m2[base_m2 + 5];
                    prod[base_prod + 5] += mul5;

                    TYPE mul6 = temp_x * m2[base_m2 + 6];
                    prod[base_prod + 6] += mul6;

                    TYPE mul7 = temp_x * m2[base_m2 + 7];
                    prod[base_prod + 7] += mul7;
                }
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    gemm_blocked_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                    (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
