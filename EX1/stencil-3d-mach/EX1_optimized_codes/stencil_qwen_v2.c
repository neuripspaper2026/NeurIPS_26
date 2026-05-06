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
    // Merge all boundary loops to reduce loop overhead
    // Process all boundaries in a single loop structure
    for (i = 0; i < height_size; i++) {
        for (j = 0; j < col_size; j++) {
            for (k = 0; k < row_size; k++) {
                if (i == 0 || i == height_size-1 || 
                    j == 0 || j == col_size-1 || 
                    k == 0 || k == row_size-1) {
                    sol[INDX(row_size, col_size, k, j, i)] = orig[INDX(row_size, col_size, k, j, i)];
                }
            }
        }
    }

    // Cache C[0] and C[1] for better performance
    const TYPE c0 = C[0];
    const TYPE c1 = C[1];

    // Stencil computation with strength reduction and loop invariant hoisting
    const int height_end = height_size - 1;
    const int col_end = col_size - 1;
    const int row_end = row_size - 1;
    
    loop_height : for(i = 1; i < height_end; i++){
        const int i_plus_1 = i + 1;
        const int i_minus_1 = i - 1;
        const int base_i = i * row_size * col_size;
        const int base_i_plus_1 = i_plus_1 * row_size * col_size;
        const int base_i_minus_1 = i_minus_1 * row_size * col_size;
        
        loop_col : for(j = 1; j < col_end; j++){
            const int j_plus_1 = j + 1;
            const int j_minus_1 = j - 1;
            const int base_ij = base_i + j * row_size;
            const int base_ij_plus_1 = base_i + j_plus_1 * row_size;
            const int base_ij_minus_1 = base_i + j_minus_1 * row_size;
            const int base_i_plus_1_j = base_i_plus_1 + j * row_size;
            const int base_i_minus_1_j = base_i_minus_1 + j * row_size;
            
            loop_row : for(k = 1; k < row_end; k++){
                sum0 = orig[base_ij + k];
                sum1 = orig[base_i_plus_1_j + k] +
                       orig[base_i_minus_1_j + k] +
                       orig[base_ij_plus_1 + k] +
                       orig[base_ij_minus_1 + k] +
                       orig[base_ij + k + 1] +
                       orig[base_ij + k - 1];
                mul0 = sum0 * c0;
                mul1 = sum1 * c1;
                sol[base_ij + k] = mul0 + mul1;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_3d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
