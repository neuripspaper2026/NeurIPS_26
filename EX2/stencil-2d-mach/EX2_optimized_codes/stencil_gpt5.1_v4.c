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
    TYPE f00, f01, f02,
         f10, f11, f12,
         f20, f21, f22;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    /* Hoist filter values into scalars to avoid repeated loads */
    f00 = filter[0]; f01 = filter[1]; f02 = filter[2];
    f10 = filter[3]; f11 = filter[4]; f12 = filter[5];
    f20 = filter[6]; f21 = filter[7]; f22 = filter[8];

    /* Parallelize outer loop over rows; collapse with columns for better balance.
       Also enable vectorization on inner dimension. */
#ifdef _OPENMP
#pragma omp parallel for collapse(2) private(r,c) schedule(static)
#endif
    for (r = 0; r < row_size - 2; r++) {
        for (c = 0; c < col_size - 2; c++) {
            int base = r * col_size + c;
            TYPE v00 = orig[base];
            TYPE v01 = orig[base + 1];
            TYPE v02 = orig[base + 2];

            TYPE v10 = orig[base + col_size];
            TYPE v11 = orig[base + col_size + 1];
            TYPE v12 = orig[base + col_size + 2];

            TYPE v20 = orig[base + 2 * col_size];
            TYPE v21 = orig[base + 2 * col_size + 1];
            TYPE v22 = orig[base + 2 * col_size + 2];

            TYPE temp =
                f00 * v00 + f01 * v01 + f02 * v02 +
                f10 * v10 + f11 * v11 + f12 * v12 +
                f20 * v20 + f21 * v21 + f22 * v22;

            sol[base] = temp;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_2d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
