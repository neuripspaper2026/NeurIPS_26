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

    // Pre-calculate filter values to avoid repeated multiplications
    const TYPE f00 = filter[0];
    const TYPE f01 = filter[1];
    const TYPE f02 = filter[2];
    const TYPE f10 = filter[3];
    const TYPE f11 = filter[4];
    const TYPE f12 = filter[5];
    const TYPE f20 = filter[6];
    const TYPE f21 = filter[7];
    const TYPE f22 = filter[8];

    stencil_label1:for (r=0; r<row_size-2; r++) {
        const int r_offset = r * col_size;
        const int r1_offset = (r+1) * col_size;
        const int r2_offset = (r+2) * col_size;
        
        stencil_label2:for (c=0; c<col_size-2; c++) {
            // Unroll the inner loops and directly access memory
            temp = f00 * orig[r_offset + c] +
                   f01 * orig[r_offset + c + 1] +
                   f02 * orig[r_offset + c + 2] +
                   f10 * orig[r1_offset + c] +
                   f11 * orig[r1_offset + c + 1] +
                   f12 * orig[r1_offset + c + 2] +
                   f20 * orig[r2_offset + c] +
                   f21 * orig[r2_offset + c + 1] +
                   f22 * orig[r2_offset + c + 2];
            
            sol[r_offset + c] = temp;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_2d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
