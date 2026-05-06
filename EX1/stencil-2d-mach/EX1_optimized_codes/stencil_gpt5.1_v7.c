#include <time.h>
#include "../stencil.h"

static double stencil_2d_kernel_time_acc = 0.0;

void reset_stencil_2d_kernel_time(void) { stencil_2d_kernel_time_acc = 0.0; }
double get_stencil_2d_kernel_time(void) { return stencil_2d_kernel_time_acc; }

void stencil (TYPE orig[row_size * col_size], TYPE sol[row_size * col_size], TYPE filter[f_size]){
    int r, c;
    struct timespec kernel_start, kernel_end;

    const TYPE f00 = filter[0];
    const TYPE f01 = filter[1];
    const TYPE f02 = filter[2];
    const TYPE f10 = filter[3];
    const TYPE f11 = filter[4];
    const TYPE f12 = filter[5];
    const TYPE f20 = filter[6];
    const TYPE f21 = filter[7];
    const TYPE f22 = filter[8];

    const int max_r = row_size - 2;
    const int max_c = col_size - 2;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    for (r = 0; r < max_r; r++) {
        int row0 = r * col_size;
        int row1 = row0 + col_size;
        int row2 = row1 + col_size;
        for (c = 0; c < max_c; c++) {
            const int idx0 = row0 + c;
            const int idx1 = row1 + c;
            const int idx2 = row2 + c;

            TYPE t0 = orig[idx0];
            TYPE t1 = orig[idx0 + 1];
            TYPE t2 = orig[idx0 + 2];
            TYPE t3 = orig[idx1];
            TYPE t4 = orig[idx1 + 1];
            TYPE t5 = orig[idx1 + 2];
            TYPE t6 = orig[idx2];
            TYPE t7 = orig[idx2 + 1];
            TYPE t8 = orig[idx2 + 2];

            TYPE temp = (TYPE)0;
            temp += f00 * t0;
            temp += f01 * t1;
            temp += f02 * t2;
            temp += f10 * t3;
            temp += f11 * t4;
            temp += f12 * t5;
            temp += f20 * t6;
            temp += f21 * t7;
            temp += f22 * t8;

            sol[idx0] = temp;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_2d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
