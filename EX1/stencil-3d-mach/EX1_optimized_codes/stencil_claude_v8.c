#include <time.h>
#include "../stencil.h"

static double stencil_3d_kernel_time_acc = 0.0;

void reset_stencil_3d_kernel_time(void) { stencil_3d_kernel_time_acc = 0.0; }
double get_stencil_3d_kernel_time(void) { return stencil_3d_kernel_time_acc; }

void stencil3d(TYPE C[2], TYPE orig[SIZE], TYPE sol[SIZE]) {
    int i, j, k;
    TYPE sum0, sum1, mul0, mul1;
    struct timespec kernel_start, kernel_end;
    TYPE c0 = C[0];
    TYPE c1 = C[1];

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    // Handle boundary conditions by filling with original values
    height_bound_col : for(j=0; j<col_size; j++) {
        height_bound_row : for(k=0; k<row_size; k++) {
            int idx_bottom = k + row_size * j;
            int idx_top = k + row_size * (j + col_size * (height_size-1));
            sol[idx_bottom] = orig[idx_bottom];
            sol[idx_top] = orig[idx_top];
        }
    }
    col_bound_height : for(i=1; i<height_size-1; i++) {
        int base_offset = col_size * i;
        col_bound_row : for(k=0; k<row_size; k++) {
            int idx_left = k + row_size * base_offset;
            int idx_right = k + row_size * ((col_size-1) + base_offset);
            sol[idx_left] = orig[idx_left];
            sol[idx_right] = orig[idx_right];
        }
    }
    row_bound_height : for(i=1; i<height_size-1; i++) {
        int base_offset = col_size * i;
        row_bound_col : for(j=1; j<col_size-1; j++) {
            int idx_front = row_size * (j + base_offset);
            int idx_back = (row_size-1) + row_size * (j + base_offset);
            sol[idx_front] = orig[idx_front];
            sol[idx_back] = orig[idx_back];
        }
    }

    // Stencil computation
    loop_height : for(i = 1; i < height_size - 1; i++){
        int i_offset = col_size * i;
        int i_plus_offset = col_size * (i + 1);
        int i_minus_offset = col_size * (i - 1);
        
        loop_col : for(j = 1; j < col_size - 1; j++){
            int base_idx = row_size * (j + i_offset);
            int j_plus_idx = row_size * ((j + 1) + i_offset);
            int j_minus_idx = row_size * ((j - 1) + i_offset);
            int i_plus_idx = row_size * (j + i_plus_offset);
            int i_minus_idx = row_size * (j + i_minus_offset);
            
            loop_row : for(k = 1; k < row_size - 1; k++){
                int center_idx = k + base_idx;
                sum0 = orig[center_idx];
                sum1 = orig[k + i_plus_idx] +
                       orig[k + i_minus_idx] +
                       orig[k + j_plus_idx] +
                       orig[k + j_minus_idx] +
                       orig[(k + 1) + base_idx] +
                       orig[(k - 1) + base_idx];
                mul0 = sum0 * c0;
                mul1 = sum1 * c1;
                sol[center_idx] = mul0 + mul1;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_3d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
