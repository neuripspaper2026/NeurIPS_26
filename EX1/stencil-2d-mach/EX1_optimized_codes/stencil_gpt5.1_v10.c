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
        int base_row0 = r * col_size;
        int base_row1 = base_row0 + col_size;
        int base_row2 = base_row1 + col_size;

        for (c = 0; c < col_size - 2; c++) {
            int idx0 = base_row0 + c;
            int idx1 = base_row1 + c;
            int idx2 = base_row2 + c;

            TYPE v00 = orig[idx0];
            TYPE v01 = orig[idx0 + 1];
            TYPE v02 = orig[idx0 + 2];
            TYPE v10 = orig[idx1];
            TYPE v11 = orig[idx1 + 1];
            TYPE v12 = orig[idx1 + 2];
            TYPE v20 = orig[idx2];
            TYPE v21 = orig[idx2 + 1];
            TYPE v22 = orig[idx2 + 2];

            TYPE temp =
                f00 * v00 + f01 * v01 + f02 * v02 +
                f10 * v10 + f11 * v11 + f12 * v12 +
                f20 * v20 + f21 * v21 + f22 * v22;

            sol[base_row0 + c] = temp;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_2d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
