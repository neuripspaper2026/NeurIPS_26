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
    // Flatten nested loops where possible and reduce INDX calls
    for(j=0; j<col_size; j++) {
        for(k=0; k<row_size; k++) {
            const int idx_bottom = INDX(row_size, col_size, k, j, 0);
            const int idx_top = INDX(row_size, col_size, k, j, height_size-1);
            sol[idx_bottom] = orig[idx_bottom];
            sol[idx_top] = orig[idx_top];
        }
    }
    
    for(i=1; i<height_size-1; i++) {
        for(k=0; k<row_size; k++) {
            const int idx_front = INDX(row_size, col_size, k, 0, i);
            const int idx_back = INDX(row_size, col_size, k, col_size-1, i);
            sol[idx_front] = orig[idx_front];
            sol[idx_back] = orig[idx_back];
        }
    }
    
    for(i=1; i<height_size-1; i++) {
        for(j=1; j<col_size-1; j++) {
            const int idx_left = INDX(row_size, col_size, 0, j, i);
            const int idx_right = INDX(row_size, col_size, row_size-1, j, i);
            sol[idx_left] = orig[idx_left];
            sol[idx_right] = orig[idx_right];
        }
    }

    // Stencil computation
    // Precompute constants and reduce redundant addressing calculations
    const int height_end = height_size - 1;
    const int col_end = col_size - 1;
    const int row_end = row_size - 1;
    
    for(i = 1; i < height_end; i++){
        for(j = 1; j < col_end; j++){
            const int base_offset = i * row_size * col_size + j * row_size;
            TYPE* orig_slice_center = &orig[base_offset];
            TYPE* orig_slice_up = &orig[base_offset + row_size * col_size];
            TYPE* orig_slice_down = &orig[base_offset - row_size * col_size];
            TYPE* orig_slice_fwd = &orig[base_offset + row_size];
            TYPE* orig_slice_bwd = &orig[base_offset - row_size];
            TYPE* sol_slice = &sol[base_offset];
            
            for(k = 1; k < row_end; k++){
                sum0 = orig_slice_center[k];
                sum1 = orig_slice_up[k] +
                       orig_slice_down[k] +
                       orig_slice_fwd[k] +
                       orig_slice_bwd[k] +
                       orig_slice_center[k + 1] +
                       orig_slice_center[k - 1];
                mul0 = sum0 * c0;
                mul1 = sum1 * c1;
                sol_slice[k] = mul0 + mul1;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_3d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
