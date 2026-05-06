#include <time.h>
#include "../stencil.h"

static double stencil_2d_kernel_time_acc = 0.0;

void reset_stencil_2d_kernel_time(void) { stencil_2d_kernel_time_acc = 0.0; }
double get_stencil_2d_kernel_time(void) { return stencil_2d_kernel_time_acc; }

void stencil (TYPE orig[row_size * col_size], TYPE sol[row_size * col_size], TYPE filter[f_size]){
    int r, c;
    TYPE f00 = filter[0];
    TYPE f01 = filter[1];
    TYPE f02 = filter[2];
    TYPE f10 = filter[3];
    TYPE f11 = filter[4];
    TYPE f12 = filter[5];
    TYPE f20 = filter[6];
    TYPE f21 = filter[7];
    TYPE f22 = filter[8];

    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    for (r = 0; r < row_size - 2; r++) {
        int base_r0 = r * col_size;
        int base_r1 = base_r0 + col_size;
        int base_r2 = base_r1 + col_size;

        for (c = 0; c < col_size - 2; c++) {
            int idx0 = base_r0 + c;
            int idx1 = base_r1 + c;
            int idx2 = base_r2 + c;

            TYPE p00 = orig[idx0];
            TYPE p01 = orig[idx0 + 1];
            TYPE p02 = orig[idx0 + 2];
            TYPE p10 = orig[idx1];
            TYPE p11 = orig[idx1 + 1];
            TYPE p12 = orig[idx1 + 2];
            TYPE p20 = orig[idx2];
            TYPE p21 = orig[idx2 + 1];
            TYPE p22 = orig[idx2 + 2];

            TYPE temp = 0;
            temp += f00 * p00;
            temp += f01 * p01;
            temp += f02 * p02;
            temp += f10 * p10;
            temp += f11 * p11;
            temp += f12 * p12;
            temp += f20 * p20;
            temp += f21 * p21;
            temp += f22 * p22;

            sol[base_r0 + c] = temp;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_2d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
