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
        TYPE *orig_r0 = &orig[r * col_size];
        TYPE *orig_r1 = &orig[(r+1) * col_size];
        TYPE *orig_r2 = &orig[(r+2) * col_size];
        TYPE *sol_r = &sol[r * col_size];
        
        stencil_label2:for (c=0; c<col_size-2; c++) {
            temp = f0 * orig_r0[c]   + f1 * orig_r0[c+1] + f2 * orig_r0[c+2] +
                   f3 * orig_r1[c]   + f4 * orig_r1[c+1] + f5 * orig_r1[c+2] +
                   f6 * orig_r2[c]   + f7 * orig_r2[c+1] + f8 * orig_r2[c+2];
            sol_r[c] = temp;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_2d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
