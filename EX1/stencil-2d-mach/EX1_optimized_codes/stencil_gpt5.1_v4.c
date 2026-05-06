#include <time.h>
#include "../stencil.h"

static double stencil_2d_kernel_time_acc = 0.0;

void reset_stencil_2d_kernel_time(void) { stencil_2d_kernel_time_acc = 0.0; }
double get_stencil_2d_kernel_time(void) { return stencil_2d_kernel_time_acc; }

void stencil (TYPE orig[row_size * col_size], TYPE sol[row_size * col_size], TYPE filter[f_size]){
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    const int inner_row_end = row_size - 2;
    const int inner_col_end = col_size - 2;

    const TYPE f00 = filter[0];
    const TYPE f01 = filter[1];
    const TYPE f02 = filter[2];
    const TYPE f10 = filter[3];
    const TYPE f11 = filter[4];
    const TYPE f12 = filter[5];
    const TYPE f20 = filter[6];
    const TYPE f21 = filter[7];
    const TYPE f22 = filter[8];

    for (int r = 0; r < inner_row_end; ++r) {
        const int row0 = r * col_size;
        const int row1 = row0 + col_size;
        const int row2 = row1 + col_size;

        for (int c = 0; c < inner_col_end; ++c) {
            const int idx0 = row0 + c;
            const int idx1 = row1 + c;
            const int idx2 = row2 + c;

            const TYPE v00 = orig[idx0];
            const TYPE v01 = orig[idx0 + 1];
            const TYPE v02 = orig[idx0 + 2];
            const TYPE v10 = orig[idx1];
            const TYPE v11 = orig[idx1 + 1];
            const TYPE v12 = orig[idx1 + 2];
            const TYPE v20 = orig[idx2];
            const TYPE v21 = orig[idx2 + 1];
            const TYPE v22 = orig[idx2 + 2];

            TYPE temp =
                f00 * v00 + f01 * v01 + f02 * v02 +
                f10 * v10 + f11 * v11 + f12 * v12 +
                f20 * v20 + f21 * v21 + f22 * v22;

            sol[idx0] = temp;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_2d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
