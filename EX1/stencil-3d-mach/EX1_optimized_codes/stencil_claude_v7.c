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
        const int i_offset = i * col_size * row_size;
        const int i_plus_offset = (i + 1) * col_size * row_size;
        const int i_minus_offset = (i - 1) * col_size * row_size;
        
        loop_col : for(j = 1; j < col_size - 1; j++){
            const int j_offset = j * row_size;
            const int j_plus_offset = (j + 1) * row_size;
            const int j_minus_offset = (j - 1) * row_size;
            
            loop_row : for(k = 1; k < row_size - 1; k++){
                const int idx_center = k + j_offset + i_offset;
                
                TYPE sum0 = orig[idx_center];
                TYPE sum1 = orig[k + j_offset + i_plus_offset] +
                           orig[k + j_offset + i_minus_offset] +
                           orig[k + j_plus_offset + i_offset] +
                           orig[k + j_minus_offset + i_offset] +
                           orig[k + 1 + j_offset + i_offset] +
                           orig[k - 1 + j_offset + i_offset];
                
                sol[idx_center] = sum0 * c0 + sum1 * c1;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_3d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
