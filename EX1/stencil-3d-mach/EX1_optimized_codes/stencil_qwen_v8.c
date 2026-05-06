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
    // Optimize by pre-computing row_size*col_size and using direct array indexing
    height_bound_col : for(j=0; j<col_size; j++) {
        const int j_times_row_size = j * row_size;
        height_bound_row : for(k=0; k<row_size; k++) {
            const int idx_base = k + j_times_row_size;
            sol[idx_base] = orig[idx_base];
            sol[idx_base + height_size_minus_1 * col_size_times_row_size] = orig[idx_base + height_size_minus_1 * col_size_times_row_size];
        }
    }
    
    col_bound_height : for(i=1; i<height_size_minus_1; i++) {
        const int i_times_col_size_times_row_size = i * col_size_times_row_size;
        const int col_size_minus_1_times_row_size = col_size_minus_1 * row_size;
        col_bound_row : for(k=0; k<row_size; k++) {
            sol[k + i_times_col_size_times_row_size] = orig[k + i_times_col_size_times_row_size];
            sol[k + col_size_minus_1_times_row_size + i_times_col_size_times_row_size] = orig[k + col_size_minus_1_times_row_size + i_times_col_size_times_row_size];
        }
    }
    
    row_bound_height : for(i=1; i<height_size_minus_1; i++) {
        const int i_times_col_size_times_row_size = i * col_size_times_row_size;
        row_bound_col : for(j=1; j<col_size_minus_1; j++) {
            const int j_times_row_size = j * row_size;
            const int idx_base = j_times_row_size + i_times_col_size_times_row_size;
            sol[idx_base] = orig[idx_base];
            sol[row_size_minus_1 + idx_base] = orig[row_size_minus_1 + idx_base];
        }
    }

    // Stencil computation
    loop_height : for(i = 1; i < height_size_minus_1; i++){
        const int i_times_col_size_times_row_size = i * col_size_times_row_size;
        const int i_plus_1_times_col_size_times_row_size = (i + 1) * col_size_times_row_size;
        const int i_minus_1_times_col_size_times_row_size = (i - 1) * col_size_times_row_size;
        
        loop_col : for(j = 1; j < col_size_minus_1; j++){
            const int j_times_row_size = j * row_size;
            const int j_plus_1_times_row_size = (j + 1) * row_size;
            const int j_minus_1_times_row_size = (j - 1) * row_size;
            
            const int idx_base = i_times_col_size_times_row_size + j_times_row_size;
            
            loop_row : for(k = 1; k < row_size_minus_1; k++){
                const int idx = idx_base + k;
                sum0 = orig[idx];
                sum1 = orig[idx + col_size_times_row_size] +  // i+1
                       orig[idx - col_size_times_row_size] +  // i-1
                       orig[idx + row_size] +                 // j+1
                       orig[idx - row_size] +                 // j-1
                       orig[idx + 1] +                        // k+1
                       orig[idx - 1];                         // k-1
                mul0 = sum0 * c0;
                mul1 = sum1 * c1;
                sol[idx] = mul0 + mul1;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_3d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
