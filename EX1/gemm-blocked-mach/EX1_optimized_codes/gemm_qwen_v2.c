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

    // Prefetch and loop unrolling optimizations
    const int row_size_unroll = row_size - (row_size % 4); // Process 4 elements at a time
    const int block_size_unroll = block_size - (block_size % 4); // Unroll inner loops if possible
    
    // Precompute common subexpressions outside inner loops where possible
    const int block_row_size = block_size * row_size;

    for (jj = 0; jj < row_size; jj += block_size){
        for (kk = 0; kk < row_size; kk += block_size){
            for ( i = 0; i < row_size; ++i){
                i_row = i * row_size;
                const int i_base = i_row + jj; // Precompute base index for prod
                for (k = 0; k < block_size; ++k){
                    k_row = (k  + kk) * row_size;
                    temp_x = m1[i_row + k + kk];
                    // Unroll the innermost loop by 4 if safe to do so
                    j = 0;
                    for (; j <= block_size - 4; j += 4){
                        mul = temp_x * m2[k_row + j + jj];
                        prod[i_base + j] += mul;
                        
                        mul = temp_x * m2[k_row + j + jj + 1];
                        prod[i_base + j + 1] += mul;
                        
                        mul = temp_x * m2[k_row + j + jj + 2];
                        prod[i_base + j + 2] += mul;
                        
                        mul = temp_x * m2[k_row + j + jj + 3];
                        prod[i_base + j + 3] += mul;
                    }
                    // Handle remaining iterations
                    for (; j < block_size; ++j){
                        mul = temp_x * m2[k_row + j + jj];
                        prod[i_base + j] += mul;
                    }
                }
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    gemm_blocked_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                    (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
