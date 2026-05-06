#include <time.h>
#include "../stencil.h"

static double stencil_3d_kernel_time_acc = 0.0;

void reset_stencil_3d_kernel_time(void) { stencil_3d_kernel_time_acc = 0.0; }
double get_stencil_3d_kernel_time(void) { return stencil_3d_kernel_time_acc; }

void stencil3d(TYPE C[2], TYPE orig[SIZE], TYPE sol[SIZE]) {
    int i, j, k;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    const TYPE c0 = C[0];
    const TYPE c1 = C[1];

    // Handle boundary conditions by filling with original values
    height_bound_col : for(j=0; j<col_size; j++) {
        height_bound_row : for(k=0; k<row_size; k++) {
            sol[INDX(row_size, col_size, k, j, 0)] = orig[INDX(row_size, col_size, k, j, 0)];
            sol[INDX(row_size, col_size, k, j, height_size-1)] = orig[INDX(row_size, col_size, k, j, height_size-1)];
        }
    }
    col_bound_height : for(i=1; i<height_size-1; i++) {
        col_bound_row : for(k=0; k<row_size; k++) {
            sol[INDX(row_size, col_size, k, 0, i)] = orig[INDX(row_size, col_size, k, 0, i)];
            sol[INDX(row_size, col_size, k, col_size-1, i)] = orig[INDX(row_size, col_size, k, col_size-1, i)];
        }
    }
    row_bound_height : for(i=1; i<height_size-1; i++) {
        row_bound_col : for(j=1; j<col_size-1; j++) {
            sol[INDX(row_size, col_size, 0, j, i)] = orig[INDX(row_size, col_size, 0, j, i)];
            sol[INDX(row_size, col_size, row_size-1, j, i)] = orig[INDX(row_size, col_size, row_size-1, j, i)];
        }
    }


    // Stencil computation
    loop_height : for(i = 1; i < height_size - 1; i++){
        const int base_i = row_size * col_size * i;
        const int base_i_minus = row_size * col_size * (i - 1);
        const int base_i_plus = row_size * col_size * (i + 1);
        
        loop_col : for(j = 1; j < col_size - 1; j++){
            const int base_j = row_size * j;
            const int base_j_minus = row_size * (j - 1);
            const int base_j_plus = row_size * (j + 1);
            
            const int idx_base = base_i + base_j;
            const int idx_i_minus = base_i_minus + base_j;
            const int idx_i_plus = base_i_plus + base_j;
            const int idx_j_minus = base_i + base_j_minus;
            const int idx_j_plus = base_i + base_j_plus;
            
            loop_row : for(k = 1; k < row_size - 1; k++){
                TYPE center = orig[idx_base + k];
                TYPE sum1 = orig[idx_i_plus + k] +
                           orig[idx_i_minus + k] +
                           orig[idx_j_plus + k] +
                           orig[idx_j_minus + k] +
                           orig[idx_base + k + 1] +
                           orig[idx_base + k - 1];
                sol[idx_base + k] = center * c0 + sum1 * c1;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_3d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
