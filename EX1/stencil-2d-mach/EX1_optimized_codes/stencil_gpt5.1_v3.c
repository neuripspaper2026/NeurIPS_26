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

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    for (r = 0; r < row_size - 2; r++) {
        const int base_r0 = r * col_size;
        const int base_r1 = (r + 1) * col_size;
        const int base_r2 = (r + 2) * col_size;

        for (c = 0; c < col_size - 2; c++) {
            const int idx00 = base_r0 + c;
            const int idx01 = idx00 + 1;
            const int idx02 = idx00 + 2;

            const int idx10 = base_r1 + c;
            const int idx11 = idx10 + 1;
            const int idx12 = idx10 + 2;

            const int idx20 = base_r2 + c;
            const int idx21 = idx20 + 1;
            const int idx22 = idx20 + 2;

            TYPE temp =
                f00 * orig[idx00] +
                f01 * orig[idx01] +
                f02 * orig[idx02] +
                f10 * orig[idx10] +
                f11 * orig[idx11] +
                f12 * orig[idx12] +
                f20 * orig[idx20] +
                f21 * orig[idx21] +
                f22 * orig[idx22];

            sol[idx00] = temp;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_2d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
