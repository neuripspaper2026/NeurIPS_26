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

    loopjj:for (jj = 0; jj < row_size; jj += block_size){
        loopkk:for (kk = 0; kk < row_size; kk += block_size){
            loopi:for ( i = 0; i < row_size; ++i){
                i_row = i * row_size;
                loopk:for (k = 0; k < block_size; ++k){
                    k_row = (k  + kk) * row_size;
                    temp_x = m1[i_row + k + kk];
                    TYPE prod_i_jj = prod[i_row + jj];
                    TYPE prod_i_jj1 = prod[i_row + jj + 1];
                    TYPE prod_i_jj2 = prod[i_row + jj + 2];
                    TYPE prod_i_jj3 = prod[i_row + jj + 3];
                    TYPE prod_i_jj4 = prod[i_row + jj + 4];
                    TYPE prod_i_jj5 = prod[i_row + jj + 5];
                    TYPE prod_i_jj6 = prod[i_row + jj + 6];
                    TYPE prod_i_jj7 = prod[i_row + jj + 7];
                    
                    TYPE m2_k_jj = m2[k_row + jj];
                    TYPE m2_k_jj1 = m2[k_row + jj + 1];
                    TYPE m2_k_jj2 = m2[k_row + jj + 2];
                    TYPE m2_k_jj3 = m2[k_row + jj + 3];
                    TYPE m2_k_jj4 = m2[k_row + jj + 4];
                    TYPE m2_k_jj5 = m2[k_row + jj + 5];
                    TYPE m2_k_jj6 = m2[k_row + jj + 6];
                    TYPE m2_k_jj7 = m2[k_row + jj + 7];
                    
                    prod_i_jj += temp_x * m2_k_jj;
                    prod_i_jj1 += temp_x * m2_k_jj1;
                    prod_i_jj2 += temp_x * m2_k_jj2;
                    prod_i_jj3 += temp_x * m2_k_jj3;
                    prod_i_jj4 += temp_x * m2_k_jj4;
                    prod_i_jj5 += temp_x * m2_k_jj5;
                    prod_i_jj6 += temp_x * m2_k_jj6;
                    prod_i_jj7 += temp_x * m2_k_jj7;
                    
                    prod[i_row + jj] = prod_i_jj;
                    prod[i_row + jj + 1] = prod_i_jj1;
                    prod[i_row + jj + 2] = prod_i_jj2;
                    prod[i_row + jj + 3] = prod_i_jj3;
                    prod[i_row + jj + 4] = prod_i_jj4;
                    prod[i_row + jj + 5] = prod_i_jj5;
                    prod[i_row + jj + 6] = prod_i_jj6;
                    prod[i_row + jj + 7] = prod_i_jj7;
                }
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    gemm_blocked_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                    (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
