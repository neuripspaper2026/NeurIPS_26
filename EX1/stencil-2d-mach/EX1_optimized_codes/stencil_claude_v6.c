#include <time.h>
#include "../stencil.h"

static double stencil_2d_kernel_time_acc = 0.0;

void reset_stencil_2d_kernel_time(void) { stencil_2d_kernel_time_acc = 0.0; }
double get_stencil_2d_kernel_time(void) { return stencil_2d_kernel_time_acc; }

void stencil (TYPE orig[row_size * col_size], TYPE sol[row_size * col_size], TYPE filter[f_size]){
    int r, c;
    TYPE temp;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    TYPE f0 = filter[0], f1 = filter[1], f2 = filter[2];
    TYPE f3 = filter[3], f4 = filter[4], f5 = filter[5];
    TYPE f6 = filter[6], f7 = filter[7], f8 = filter[8];

    stencil_label1:for (r=0; r<row_size-2; r++) {
        int row0_base = r * col_size;
        int row1_base = (r + 1) * col_size;
        int row2_base = (r + 2) * col_size;
        int sol_base = r * col_size;
        
        stencil_label2:for (c=0; c<col_size-2; c++) {
            TYPE o00 = orig[row0_base + c];
            TYPE o01 = orig[row0_base + c + 1];
            TYPE o02 = orig[row0_base + c + 2];
            
            TYPE o10 = orig[row1_base + c];
            TYPE o11 = orig[row1_base + c + 1];
            TYPE o12 = orig[row1_base + c + 2];
            
            TYPE o20 = orig[row2_base + c];
            TYPE o21 = orig[row2_base + c + 1];
            TYPE o22 = orig[row2_base + c + 2];
            
            temp = f0 * o00 + f1 * o01 + f2 * o02 +
                   f3 * o10 + f4 * o11 + f5 * o12 +
                   f6 * o20 + f7 * o21 + f8 * o22;
            
            sol[sol_base + c] = temp;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_2d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
