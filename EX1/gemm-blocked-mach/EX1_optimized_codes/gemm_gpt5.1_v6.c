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

    for (jj = 0; jj < row_size; jj += block_size){
        for (kk = 0; kk < row_size; kk += block_size){
            for ( i = 0; i < row_size; ++i){
                i_row = i * row_size;
                for (k = 0; k < block_size; ++k){
                    int k_index = k + kk;
                    k_row = k_index * row_size;
                    temp_x = m1[i_row + k_index];

                    /* Manually unroll inner j-loop for block_size == 8 */
                    j = 0;
                    mul = temp_x * m2[k_row + jj + 0];
                    prod[i_row + jj + 0] += mul;

                    mul = temp_x * m2[k_row + jj + 1];
                    prod[i_row + jj + 1] += mul;

                    mul = temp_x * m2[k_row + jj + 2];
                    prod[i_row + jj + 2] += mul;

                    mul = temp_x * m2[k_row + jj + 3];
                    prod[i_row + jj + 3] += mul;

                    mul = temp_x * m2[k_row + jj + 4];
                    prod[i_row + jj + 4] += mul;

                    mul = temp_x * m2[k_row + jj + 5];
                    prod[i_row + jj + 5] += mul;

                    mul = temp_x * m2[k_row + jj + 6];
                    prod[i_row + jj + 6] += mul;

                    mul = temp_x * m2[k_row + jj + 7];
                    prod[i_row + jj + 7] += mul;
                }
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    gemm_blocked_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                    (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
