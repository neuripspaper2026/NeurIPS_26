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
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    #ifdef _OPENMP
    #pragma omp parallel for private(c, k1, k2) schedule(static)
    #endif
    stencil_label1:for (r=0; r<row_size-2; r++) {
        stencil_label2:for (c=0; c<col_size-2; c++) {
            TYPE temp = (TYPE)0;
            TYPE f0 = filter[0], f1 = filter[1], f2 = filter[2];
            TYPE f3 = filter[3], f4 = filter[4], f5 = filter[5];
            TYPE f6 = filter[6], f7 = filter[7], f8 = filter[8];
            
            int base_idx = r * col_size + c;
            int row1_idx = base_idx;
            int row2_idx = base_idx + col_size;
            int row3_idx = base_idx + 2 * col_size;
            
            temp += f0 * orig[row1_idx];
            temp += f1 * orig[row1_idx + 1];
            temp += f2 * orig[row1_idx + 2];
            
            temp += f3 * orig[row2_idx];
            temp += f4 * orig[row2_idx + 1];
            temp += f5 * orig[row2_idx + 2];
            
            temp += f6 * orig[row3_idx];
            temp += f7 * orig[row3_idx + 1];
            temp += f8 * orig[row3_idx + 2];
            
            sol[r * col_size + c] = temp;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_2d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
