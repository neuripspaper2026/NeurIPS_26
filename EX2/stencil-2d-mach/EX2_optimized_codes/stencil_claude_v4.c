#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../stencil.h"

static double stencil_2d_kernel_time_acc = 0.0;

void reset_stencil_2d_kernel_time(void) { stencil_2d_kernel_time_acc = 0.0; }
double get_stencil_2d_kernel_time(void) { return stencil_2d_kernel_time_acc; }

void stencil (TYPE orig[row_size * col_size], TYPE sol[row_size * col_size], TYPE filter[f_size]){
    int r, c, k1, k2;
    TYPE temp, mul;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    // Precompute filter values to avoid repeated indexing
    TYPE f[9];
    for (k1 = 0; k1 < 9; k1++) {
        f[k1] = filter[k1];
    }

    #pragma omp parallel for private(r, c, k1, k2, temp, mul) schedule(static) if(row_size > 16)
    stencil_label1:for (r=0; r<row_size-2; r++) {
        stencil_label2:for (c=0; c<col_size-2; c++) {
            // Manual unrolling of 3x3 stencil
            int base_idx = r * col_size + c;
            int row0 = base_idx;
            int row1 = base_idx + col_size;
            int row2 = base_idx + 2 * col_size;
            
            temp = f[0] * orig[row0] + 
                   f[1] * orig[row0 + 1] + 
                   f[2] * orig[row0 + 2] +
                   f[3] * orig[row1] + 
                   f[4] * orig[row1 + 1] + 
                   f[5] * orig[row1 + 2] +
                   f[6] * orig[row2] + 
                   f[7] * orig[row2 + 1] + 
                   f[8] * orig[row2 + 2];
            
            sol[r * col_size + c] = temp;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_2d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
