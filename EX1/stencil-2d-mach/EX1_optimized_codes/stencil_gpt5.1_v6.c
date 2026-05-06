#include <time.h>
#include "../stencil.h"

static double stencil_2d_kernel_time_acc = 0.0;

void reset_stencil_2d_kernel_time(void) { stencil_2d_kernel_time_acc = 0.0; }
double get_stencil_2d_kernel_time(void) { return stencil_2d_kernel_time_acc; }

void stencil(TYPE orig[row_size * col_size], TYPE sol[row_size * col_size], TYPE filter[f_size]) {
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    /* Preload filter values into locals to help the compiler keep them in registers */
    const TYPE f0 = filter[0];
    const TYPE f1 = filter[1];
    const TYPE f2 = filter[2];
    const TYPE f3 = filter[3];
    const TYPE f4 = filter[4];
    const TYPE f5 = filter[5];
    const TYPE f6 = filter[6];
    const TYPE f7 = filter[7];
    const TYPE f8 = filter[8];

    for (int r = 0; r < row_size - 2; ++r) {
        const int base_r0 = r * col_size;
        const int base_r1 = base_r0 + col_size;
        const int base_r2 = base_r1 + col_size;

        for (int c = 0; c < col_size - 2; ++c) {
            const int idx00 = base_r0 + c;
            const int idx01 = idx00 + 1;
            const int idx02 = idx00 + 2;

            const int idx10 = base_r1 + c;
            const int idx11 = idx10 + 1;
            const int idx12 = idx10 + 2;

            const int idx20 = base_r2 + c;
            const int idx21 = idx20 + 1;
            const int idx22 = idx20 + 2;

            const TYPE v00 = orig[idx00];
            const TYPE v01 = orig[idx01];
            const TYPE v02 = orig[idx02];

            const TYPE v10 = orig[idx10];
            const TYPE v11 = orig[idx11];
            const TYPE v12 = orig[idx12];

            const TYPE v20 = orig[idx20];
            const TYPE v21 = orig[idx21];
            const TYPE v22 = orig[idx22];

            TYPE temp =
                f0 * v00 + f1 * v01 + f2 * v02 +
                f3 * v10 + f4 * v11 + f5 * v12 +
                f6 * v20 + f7 * v21 + f8 * v22;

            sol[base_r0 + c] = temp;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_2d_kernel_time_acc +=
        (kernel_end.tv_sec - kernel_start.tv_sec) +
        (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
