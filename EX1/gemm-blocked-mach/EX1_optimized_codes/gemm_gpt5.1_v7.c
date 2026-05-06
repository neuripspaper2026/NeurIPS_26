#include <time.h>
#include "../gemm.h"

static double gemm_blocked_kernel_time_acc = 0.0;

void reset_gemm_blocked_kernel_time(void) { gemm_blocked_kernel_time_acc = 0.0; }
double get_gemm_blocked_kernel_time(void) { return gemm_blocked_kernel_time_acc; }

void bbgemm(TYPE m1[N], TYPE m2[N], TYPE prod[N]){
    int i, k, j, jj, kk;
    int i_row, k_row;
    TYPE temp_x;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    for (jj = 0; jj < row_size; jj += block_size) {
        for (kk = 0; kk < row_size; kk += block_size) {
            for (i = 0; i < row_size; ++i) {
                i_row = i * row_size;
                for (k = 0; k < block_size; ++k) {
                    int k_block = k + kk;
                    k_row = k_block * row_size;
                    temp_x = m1[i_row + k_block];

                    int base_m2 = k_row + jj;
                    int base_p  = i_row + jj;

                    int j0 = base_p;
                    int j1 = base_p + 1;
                    int j2 = base_p + 2;
                    int j3 = base_p + 3;
                    int j4 = base_p + 4;
                    int j5 = base_p + 5;
                    int j6 = base_p + 6;
                    int j7 = base_p + 7;

                    prod[j0] += temp_x * m2[base_m2    ];
                    prod[j1] += temp_x * m2[base_m2 + 1];
                    prod[j2] += temp_x * m2[base_m2 + 2];
                    prod[j3] += temp_x * m2[base_m2 + 3];
                    prod[j4] += temp_x * m2[base_m2 + 4];
                    prod[j5] += temp_x * m2[base_m2 + 5];
                    prod[j6] += temp_x * m2[base_m2 + 6];
                    prod[j7] += temp_x * m2[base_m2 + 7];
                }
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    gemm_blocked_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                    (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
