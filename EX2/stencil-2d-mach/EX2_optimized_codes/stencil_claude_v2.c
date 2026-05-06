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

    #ifdef _OPENMP
    #pragma omp parallel for private(r, c, k1, k2, temp, mul) schedule(static)
    #endif
    for (r=0; r<row_size-2; r++) {
        for (c=0; c<col_size-2; c++) {
            TYPE f0 = filter[0];
            TYPE f1 = filter[1];
            TYPE f2 = filter[2];
            TYPE f3 = filter[3];
            TYPE f4 = filter[4];
            TYPE f5 = filter[5];
            TYPE f6 = filter[6];
            TYPE f7 = filter[7];
            TYPE f8 = filter[8];
            
            int base_idx = r * col_size + c;
            int row1_idx = base_idx + col_size;
            int row2_idx = row1_idx + col_size;
            
            TYPE o0 = orig[base_idx];
            TYPE o1 = orig[base_idx + 1];
            TYPE o2 = orig[base_idx + 2];
            TYPE o3 = orig[row1_idx];
            TYPE o4 = orig[row1_idx + 1];
            TYPE o5 = orig[row1_idx + 2];
            TYPE o6 = orig[row2_idx];
            TYPE o7 = orig[row2_idx + 1];
            TYPE o8 = orig[row2_idx + 2];
            
            temp = f0 * o0 + f1 * o1 + f2 * o2 +
                   f3 * o3 + f4 * o4 + f5 * o5 +
                   f6 * o6 + f7 * o7 + f8 * o8;
            
            sol[r * col_size + c] = temp;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_2d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
