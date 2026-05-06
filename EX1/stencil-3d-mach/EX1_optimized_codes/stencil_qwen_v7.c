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
    
    const int height_size_minus_1 = height_size - 1;
    const int col_size_minus_1 = col_size - 1;
    const int row_size_minus_1 = row_size - 1;
    
    const int col_size_times_row_size = col_size * row_size;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    // Handle boundary conditions by filling with original values
    // Merge height_bound_col and height_bound_row loops
    for(j=0; j<col_size; j++) {
        for(k=0; k<row_size; k++) {
            sol[k + row_size*(j + col_size*0)] = orig[k + row_size*(j + col_size*0)];
            sol[k + row_size*(j + col_size*height_size_minus_1)] = orig[k + row_size*(j + col_size*height_size_minus_1)];
        }
    }
    
    // Merge col_bound_height and col_bound_row loops
    for(i=1; i<height_size_minus_1; i++) {
        for(k=0; k<row_size; k++) {
            sol[k + row_size*(0 + col_size*i)] = orig[k + row_size*(0 + col_size*i)];
            sol[k + row_size*(col_size_minus_1 + col_size*i)] = orig[k + row_size*(col_size_minus_1 + col_size*i)];
        }
    }
    
    // Merge row_bound_height and row_bound_col loops
    for(i=1; i<height_size_minus_1; i++) {
        for(j=1; j<col_size_minus_1; j++) {
            sol[0 + row_size*(j + col_size*i)] = orig[0 + row_size*(j + col_size*i)];
            sol[row_size_minus_1 + row_size*(j + col_size*i)] = orig[row_size_minus_1 + row_size*(j + col_size*i)];
        }
    }

    // Stencil computation
    // Precompute indices to reduce INDX calls
    for(i = 1; i < height_size_minus_1; i++){
        const int i_plus_1 = i + 1;
        const int i_minus_1 = i - 1;
        const int i_times_col_size_row_size = i * col_size_times_row_size;
        const int i_plus_1_times_col_size_row_size = i_plus_1 * col_size_times_row_size;
        const int i_minus_1_times_col_size_row_size = i_minus_1 * col_size_times_row_size;
        
        for(j = 1; j < col_size_minus_1; j++){
            const int j_plus_1 = j + 1;
            const int j_minus_1 = j - 1;
            const int j_times_row_size = j * row_size;
            const int j_plus_1_times_row_size = j_plus_1 * row_size;
            const int j_minus_1_times_row_size = j_minus_1 * row_size;
            
            for(k = 1; k < row_size_minus_1; k++){
                const int base_index = k + j_times_row_size + i_times_col_size_row_size;
                sum0 = orig[base_index];
                sum1 = orig[k + j_times_row_size + i_plus_1_times_col_size_row_size] +
                       orig[k + j_times_row_size + i_minus_1_times_col_size_row_size] +
                       orig[k + j_plus_1_times_row_size + i_times_col_size_row_size] +
                       orig[k + j_minus_1_times_row_size + i_times_col_size_row_size] +
                       orig[k + 1 + j_times_row_size + i_times_col_size_row_size] +
                       orig[k - 1 + j_times_row_size + i_times_col_size_row_size];
                mul0 = sum0 * c0;
                mul1 = sum1 * c1;
                sol[base_index] = mul0 + mul1;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_3d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
