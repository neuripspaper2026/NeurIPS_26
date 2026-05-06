#include <time.h>
#include "../stencil.h"

static double stencil_2d_kernel_time_acc = 0.0;

void reset_stencil_2d_kernel_time(void) { stencil_2d_kernel_time_acc = 0.0; }
double get_stencil_2d_kernel_time(void) { return stencil_2d_kernel_time_acc; }

void stencil (TYPE orig[row_size * col_size], TYPE sol[row_size * col_size], TYPE filter[f_size]){
    int r, c;
    TYPE f00, f01, f02, f10, f11, f12, f20, f21, f22;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    f00 = filter[0];
    f01 = filter[1];
    f02 = filter[2];
    f10 = filter[3];
    f11 = filter[4];
    f12 = filter[5];
    f20 = filter[6];
    f21 = filter[7];
    f22 = filter[8];

    for (r = 0; r < row_size - 2; r++) {
        int base_r0 = r * col_size;
        int base_r1 = (r + 1) * col_size;
        int base_r2 = (r + 2) * col_size;

        for (c = 0; c < col_size - 2; c++) {
            TYPE t0, t1, t2;
            TYPE v00 = orig[base_r0 + c];
            TYPE v01 = orig[base_r0 + c + 1];
            TYPE v02 = orig[base_r0 + c + 2];
            TYPE v10 = orig[base_r1 + c];
            TYPE v11 = orig[base_r1 + c + 1];
            TYPE v12 = orig[base_r1 + c + 2];
            TYPE v20 = orig[base_r2 + c];
            TYPE v21 = orig[base_r2 + c + 1];
            TYPE v22 = orig[base_r2 + c + 2];

            t0 = v00 * f00 + v01 * f01 + v02 * f02;
            t1 = v10 * f10 + v11 * f11 + v12 * f12;
            t2 = v20 * f20 + v21 * f21 + v22 * f22;

            sol[base_r0 + c] = t0 + t1 + t2;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_2d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
