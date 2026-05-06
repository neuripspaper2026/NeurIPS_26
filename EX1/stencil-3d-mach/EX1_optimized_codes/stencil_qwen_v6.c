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

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    // Handle boundary conditions by filling with original values
    // Process all boundary conditions in a single loop to improve cache locality
    for(j=0; j<col_size; j++) {
        for(k=0; k<row_size; k++) {
            sol[INDX(row_size, col_size, k, j, 0)] = orig[INDX(row_size, col_size, k, j, 0)];
            sol[INDX(row_size, col_size, k, j, height_size-1)] = orig[INDX(row_size, col_size, k, j, height_size-1)];
        }
    }
    
    for(i=1; i<height_size-1; i++) {
        for(k=0; k<row_size; k++) {
            sol[INDX(row_size, col_size, k, 0, i)] = orig[INDX(row_size, col_size, k, 0, i)];
            sol[INDX(row_size, col_size, k, col_size-1, i)] = orig[INDX(row_size, col_size, k, col_size-1, i)];
        }
    }
    
    for(i=1; i<height_size-1; i++) {
        for(j=1; j<col_size-1; j++) {
            sol[INDX(row_size, col_size, 0, j, i)] = orig[INDX(row_size, col_size, 0, j, i)];
            sol[INDX(row_size, col_size, row_size-1, j, i)] = orig[INDX(row_size, col_size, row_size-1, j, i)];
        }
    }

    // Stencil computation
    // Precompute loop bounds to avoid repeated calculations
    const int height_end = height_size - 1;
    const int col_end = col_size - 1;
    const int row_end = row_size - 1;
    
    for(i = 1; i < height_end; i++){
        const int i_plus_1 = i + 1;
        const int i_minus_1 = i - 1;
        TYPE *orig_slice_current = &orig[INDX(row_size, col_size, 0, 0, i)];
        TYPE *orig_slice_plus = &orig[INDX(row_size, col_size, 0, 0, i_plus_1)];
        TYPE *orig_slice_minus = &orig[INDX(row_size, col_size, 0, 0, i_minus_1)];
        TYPE *sol_slice = &sol[INDX(row_size, col_size, 0, 0, i)];
        
        for(j = 1; j < col_end; j++){
            const int j_plus_1 = j + 1;
            const int j_minus_1 = j - 1;
            
            for(k = 1; k < row_end; k++){
                const int k_plus_1 = k + 1;
                const int k_minus_1 = k - 1;
                
                // Use local pointers to avoid repeated INDX calculations
                const int current_idx = INDX(row_size, col_size, k, j, 0); // 0 offset as we already offset the slices
                sum0 = orig_slice_current[current_idx];
                sum1 = orig_slice_plus[current_idx] +
                       orig_slice_minus[current_idx] +
                       orig[INDX(row_size, col_size, k, j_plus_1, i)] +
                       orig[INDX(row_size, col_size, k, j_minus_1, i)] +
                       orig[INDX(row_size, col_size, k_plus_1, j, i)] +
                       orig[INDX(row_size, col_size, k_minus_1, j, i)];
                mul0 = sum0 * c0;
                mul1 = sum1 * c1;
                sol_slice[current_idx] = mul0 + mul1;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_3d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
