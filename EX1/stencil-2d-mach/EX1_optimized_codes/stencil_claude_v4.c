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

    TYPE f0 = filter[0];
    TYPE f1 = filter[1];
    TYPE f2 = filter[2];
    TYPE f3 = filter[3];
    TYPE f4 = filter[4];
    TYPE f5 = filter[5];
    TYPE f6 = filter[6];
    TYPE f7 = filter[7];
    TYPE f8 = filter[8];

    stencil_label1:for (r=0; r<row_size-2; r++) {
        int row0 = r * col_size;
        int row1 = (r + 1) * col_size;
        int row2 = (r + 2) * col_size;
        int sol_row = r * col_size;
        
        stencil_label2:for (c=0; c<col_size-2; c++) {
            TYPE o0 = orig[row0 + c];
            TYPE o1 = orig[row0 + c + 1];
            TYPE o2 = orig[row0 + c + 2];
            TYPE o3 = orig[row1 + c];
            TYPE o4 = orig[row1 + c + 1];
            TYPE o5 = orig[row1 + c + 2];
            TYPE o6 = orig[row2 + c];
            TYPE o7 = orig[row2 + c + 1];
            TYPE o8 = orig[row2 + c + 2];
            
            temp = f0 * o0 + f1 * o1 + f2 * o2 +
                   f3 * o3 + f4 * o4 + f5 * o5 +
                   f6 * o6 + f7 * o7 + f8 * o8;
            
            sol[sol_row + c] = temp;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_2d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
