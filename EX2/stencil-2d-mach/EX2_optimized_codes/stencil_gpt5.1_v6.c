#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../stencil.h"

static double stencil_2d_kernel_time_acc = 0.0;

void reset_stencil_2d_kernel_time(void) { stencil_2d_kernel_time_acc = 0.0; }
double get_stencil_2d_kernel_time(void) { return stencil_2d_kernel_time_acc; }

void stencil (TYPE orig[row_size * col_size], TYPE sol[row_size * col_size], TYPE filter[f_size]){
    int r, c;
    TYPE f0, f1, f2, f3, f4, f5, f6, f7, f8;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    /* Cache filter values in registers (serial optimization) */
    f0 = filter[0];
    f1 = filter[1];
    f2 = filter[2];
    f3 = filter[3];
    f4 = filter[4];
    f5 = filter[5];
    f6 = filter[6];
    f7 = filter[7];
    f8 = filter[8];

    /* Parallelize the outer loop with OpenMP, preserve timing and API */
#ifdef _OPENMP
#pragma omp parallel for private(c) schedule(static)
#endif
    for (r = 0; r < row_size - 2; r++) {
        int base_r0 = r * col_size;
        int base_r1 = (r + 1) * col_size;
        int base_r2 = (r + 2) * col_size;

        for (c = 0; c < col_size - 2; c++) {
            int idx0 = base_r0 + c;
            int idx1 = base_r1 + c;
            int idx2 = base_r2 + c;

            TYPE o00 = orig[idx0];
            TYPE o01 = orig[idx0 + 1];
            TYPE o02 = orig[idx0 + 2];

            TYPE o10 = orig[idx1];
            TYPE o11 = orig[idx1 + 1];
            TYPE o12 = orig[idx1 + 2];

            TYPE o20 = orig[idx2];
            TYPE o21 = orig[idx2 + 1];
            TYPE o22 = orig[idx2 + 2];

            TYPE temp =
                f0 * o00 + f1 * o01 + f2 * o02 +
                f3 * o10 + f4 * o11 + f5 * o12 +
                f6 * o20 + f7 * o21 + f8 * o22;

            sol[base_r0 + c] = temp;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_2d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
