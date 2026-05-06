#include <time.h>
#include "../stencil.h"

static double stencil_2d_kernel_time_acc = 0.0;

void reset_stencil_2d_kernel_time(void) { stencil_2d_kernel_time_acc = 0.0; }
double get_stencil_2d_kernel_time(void) { return stencil_2d_kernel_time_acc; }

void stencil (TYPE orig[row_size * col_size], TYPE sol[row_size * col_size], TYPE filter[f_size]){
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    const int rows_limit = row_size - 2;
    const int cols_limit = col_size - 2;
    const int stride     = col_size;

    const TYPE f00 = filter[0];
    const TYPE f01 = filter[1];
    const TYPE f02 = filter[2];
    const TYPE f10 = filter[3];
    const TYPE f11 = filter[4];
    const TYPE f12 = filter[5];
    const TYPE f20 = filter[6];
    const TYPE f21 = filter[7];
    const TYPE f22 = filter[8];

    for (int r = 0; r < rows_limit; ++r) {
        const int row0 = r * stride;
        const int row1 = row0 + stride;
        const int row2 = row1 + stride;
        int out_idx = row0;

        for (int c = 0; c < cols_limit; ++c, ++out_idx) {
            const int c1 = c + 1;
            const int c2 = c + 2;

            const TYPE v00 = orig[row0 + c];
            const TYPE v01 = orig[row0 + c1];
            const TYPE v02 = orig[row0 + c2];

            const TYPE v10 = orig[row1 + c];
            const TYPE v11 = orig[row1 + c1];
            const TYPE v12 = orig[row1 + c2];

            const TYPE v20 = orig[row2 + c];
            const TYPE v21 = orig[row2 + c1];
            const TYPE v22 = orig[row2 + c2];

            TYPE temp =
                f00 * v00 + f01 * v01 + f02 * v02 +
                f10 * v10 + f11 * v11 + f12 * v12 +
                f20 * v20 + f21 * v21 + f22 * v22;

            sol[out_idx] = temp;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_2d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
