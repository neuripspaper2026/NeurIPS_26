#include <time.h>
#include "../stencil.h"

static double stencil_3d_kernel_time_acc = 0.0;

void reset_stencil_3d_kernel_time(void) { stencil_3d_kernel_time_acc = 0.0; }
double get_stencil_3d_kernel_time(void) { return stencil_3d_kernel_time_acc; }

void stencil3d(TYPE C[2], TYPE orig[SIZE], TYPE sol[SIZE]) {
    int i, j, k;
    TYPE sum0, sum1, mul0, mul1;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    // Handle boundary conditions by filling with original values
    // Optimize boundary loops by reducing function calls and improving cache access
    const int last_height = height_size - 1;
    const int last_col = col_size - 1;
    const int last_row = row_size - 1;

    // Combine the first two boundary loops that share similar access patterns
    for(j=0; j<col_size; j++) {
        for(k=0; k<row_size; k++) {
            sol[INDX(row_size, col_size, k, j, 0)] = orig[INDX(row_size, col_size, k, j, 0)];
            sol[INDX(row_size, col_size, k, j, last_height)] = orig[INDX(row_size, col_size, k, j, last_height)];
        }
    }
    
    for(i=1; i<height_size-1; i++) {
        for(k=0; k<row_size; k++) {
            sol[INDX(row_size, col_size, k, 0, i)] = orig[INDX(row_size, col_size, k, 0, i)];
            sol[INDX(row_size, col_size, k, last_col, i)] = orig[INDX(row_size, col_size, k, last_col, i)];
        }
    }
    
    for(i=1; i<height_size-1; i++) {
        for(j=1; j<col_size-1; j++) {
            sol[INDX(row_size, col_size, 0, j, i)] = orig[INDX(row_size, col_size, 0, j, i)];
            sol[INDX(row_size, col_size, last_row, j, i)] = orig[INDX(row_size, col_size, last_row, j, i)];
        }
    }

    // Precompute coefficients to reduce repeated memory accesses
    const TYPE c0 = C[0];
    const TYPE c1 = C[1];

    // Stencil computation with optimized indexing
    loop_height : for(i = 1; i < last_height; i++){
        loop_col : for(j = 1; j < last_col; j++){
            // Precompute base index for current j,i to reduce redundant calculations
            const int base_offset = INDX(row_size, col_size, 0, j, i);
            const int up_offset = INDX(row_size, col_size, 0, j, i + 1);
            const int down_offset = INDX(row_size, col_size, 0, j, i - 1);
            const int right_offset = INDX(row_size, col_size, 0, j + 1, i);
            const int left_offset = INDX(row_size, col_size, 0, j - 1, i);
            
            loop_row : for(k = 1; k < last_row; k++){
                // Direct array access instead of repeated INDX macro calls
                sum0 = orig[base_offset + k];
                sum1 = orig[up_offset + k] +
                       orig[down_offset + k] +
                       orig[right_offset + k] +
                       orig[left_offset + k] +
                       orig[base_offset + k + 1] +
                       orig[base_offset + k - 1];
                mul0 = sum0 * c0;
                mul1 = sum1 * c1;
                sol[base_offset + k] = mul0 + mul1;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_3d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
