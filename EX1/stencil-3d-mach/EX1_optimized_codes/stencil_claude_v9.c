#include <time.h>
#include "../stencil.h"

static double stencil_3d_kernel_time_acc = 0.0;

void reset_stencil_3d_kernel_time(void) { stencil_3d_kernel_time_acc = 0.0; }
double get_stencil_3d_kernel_time(void) { return stencil_3d_kernel_time_acc; }

void stencil3d(TYPE C[2], TYPE orig[SIZE], TYPE sol[SIZE]) {
    int i, j, k;
    TYPE sum0, sum1, mul0, mul1;
    struct timespec kernel_start, kernel_end;
    
    const TYPE c0 = C[0];
    const TYPE c1 = C[1];
    const int rs = row_size;
    const int cs = col_size;
    const int hs = height_size;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    // Handle boundary conditions by filling with original values
    height_bound_col : for(j=0; j<cs; j++) {
        int base_j = rs * j;
        height_bound_row : for(k=0; k<rs; k++) {
            int idx_base = k + base_j;
            sol[idx_base] = orig[idx_base];
            sol[idx_base + rs * cs * (hs-1)] = orig[idx_base + rs * cs * (hs-1)];
        }
    }
    col_bound_height : for(i=1; i<hs-1; i++) {
        int base_i = rs * cs * i;
        col_bound_row : for(k=0; k<rs; k++) {
            sol[k + base_i] = orig[k + base_i];
            sol[k + rs * (cs-1) + base_i] = orig[k + rs * (cs-1) + base_i];
        }
    }
    row_bound_height : for(i=1; i<hs-1; i++) {
        int base_i = rs * cs * i;
        row_bound_col : for(j=1; j<cs-1; j++) {
            int idx = rs * j + base_i;
            sol[idx] = orig[idx];
            sol[rs-1 + idx] = orig[rs-1 + idx];
        }
    }

    // Stencil computation
    loop_height : for(i = 1; i < hs - 1; i++){
        int base_i = rs * cs * i;
        int base_i_prev = rs * cs * (i - 1);
        int base_i_next = rs * cs * (i + 1);
        
        loop_col : for(j = 1; j < cs - 1; j++){
            int base_j = rs * j;
            int base_j_prev = rs * (j - 1);
            int base_j_next = rs * (j + 1);
            
            int idx_center = base_j + base_i;
            int idx_i_prev = base_j + base_i_prev;
            int idx_i_next = base_j + base_i_next;
            int idx_j_prev = base_j_prev + base_i;
            int idx_j_next = base_j_next + base_i;
            
            loop_row : for(k = 1; k < rs - 1; k++){
                sum0 = orig[k + idx_center];
                sum1 = orig[k + idx_i_next] +
                       orig[k + idx_i_prev] +
                       orig[k + idx_j_next] +
                       orig[k + idx_j_prev] +
                       orig[k + 1 + idx_center] +
                       orig[k - 1 + idx_center];
                mul0 = sum0 * c0;
                mul1 = sum1 * c1;
                sol[k + idx_center] = mul0 + mul1;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_3d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
