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
    // Merge loops to reduce loop overhead
    for (j = 0; j < col_size; j++) {
        for (k = 0; k < row_size; k++) {
            const int idx_bottom = INDX(row_size, col_size, k, j, 0);
            const int idx_top = INDX(row_size, col_size, k, j, height_size-1);
            sol[idx_bottom] = orig[idx_bottom];
            sol[idx_top] = orig[idx_top];
        }
    }

    for (i = 1; i < height_size-1; i++) {
        // Process column boundaries
        for (k = 0; k < row_size; k++) {
            const int idx_front = INDX(row_size, col_size, k, 0, i);
            const int idx_back = INDX(row_size, col_size, k, col_size-1, i);
            sol[idx_front] = orig[idx_front];
            sol[idx_back] = orig[idx_back];
        }
        
        // Process row boundaries
        for (j = 1; j < col_size-1; j++) {
            const int idx_left = INDX(row_size, col_size, 0, j, i);
            const int idx_right = INDX(row_size, col_size, row_size-1, j, i);
            sol[idx_left] = orig[idx_left];
            sol[idx_right] = orig[idx_right];
        }
    }

    // Stencil computation
    // Precompute constants to reduce redundant calculations
    const int height_end = height_size - 1;
    const int col_end = col_size - 1;
    const int row_end = row_size - 1;
    
    for (i = 1; i < height_end; i++) {
        for (j = 1; j < col_end; j++) {
            // Precompute base indices to reduce redundant INDX calls
            const int base_ij = i * col_size * row_size + j * row_size;
            const int base_ij_plus_1 = (i + 1) * col_size * row_size + j * row_size;
            const int base_ij_minus_1 = (i - 1) * col_size * row_size + j * row_size;
            const int base_i_j_plus_1 = i * col_size * row_size + (j + 1) * row_size;
            const int base_i_j_minus_1 = i * col_size * row_size + (j - 1) * row_size;
            
            for (k = 1; k < row_end; k++) {
                sum0 = orig[base_ij + k];
                sum1 = orig[base_ij_plus_1 + k] +
                       orig[base_ij_minus_1 + k] +
                       orig[base_i_j_plus_1 + k] +
                       orig[base_i_j_minus_1 + k] +
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
