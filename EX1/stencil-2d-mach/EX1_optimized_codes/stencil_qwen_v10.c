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

    // Unroll the innermost loops (k1 and k2) to eliminate loop overhead
    stencil_label1:for (r=0; r<row_size-2; r++) {
        stencil_label2:for (c=0; c<col_size-2; c++) {
            // Manually unroll the 3x3 filter convolution
            temp = filter[0] * orig[r*col_size + c] +
                   filter[1] * orig[r*col_size + c+1] +
                   filter[2] * orig[r*col_size + c+2] +
                   filter[3] * orig[(r+1)*col_size + c] +
                   filter[4] * orig[(r+1)*col_size + c+1] +
                   filter[5] * orig[(r+1)*col_size + c+2] +
                   filter[6] * orig[(r+2)*col_size + c] +
                   filter[7] * orig[(r+2)*col_size + c+1] +
                   filter[8] * orig[(r+2)*col_size + c+2];
            
            sol[(r*col_size) + c] = temp;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_2d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
