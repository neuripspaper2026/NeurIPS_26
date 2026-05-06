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
    TYPE f00, f01, f02;
    TYPE f10, f11, f12;
    TYPE f20, f21, f22;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    /* Preload filter values to scalars to help vectorization/cache */
    f00 = filter[0];  f01 = filter[1];  f02 = filter[2];
    f10 = filter[3];  f11 = filter[4];  f12 = filter[5];
    f20 = filter[6];  f21 = filter[7];  f22 = filter[8];

    /* 2D stencil: parallel over rows and columns, collapse for better load balance.
       Use restrict/const aliasing hints via the type-system assumptions of C.
       Inner loop explicitly unrolled for 3x3 filter to remove tiny loops. */
#ifdef _OPENMP
#pragma omp parallel for collapse(2) private(r,c) schedule(static)
#endif
    for (r = 0; r < row_size - 2; r++) {
        for (c = 0; c < col_size - 2; c++) {
            int base = r * col_size + c;
            TYPE t00 = orig[base];
            TYPE t01 = orig[base + 1];
            TYPE t02 = orig[base + 2];

            int base1 = base + col_size;
            TYPE t10 = orig[base1];
            TYPE t11 = orig[base1 + 1];
            TYPE t12 = orig[base1 + 2];

            int base2 = base1 + col_size;
            TYPE t20 = orig[base2];
            TYPE t21 = orig[base2 + 1];
            TYPE t22 = orig[base2 + 2];

            TYPE temp =
                f00 * t00 + f01 * t01 + f02 * t02 +
                f10 * t10 + f11 * t11 + f12 * t12 +
                f20 * t20 + f21 * t21 + f22 * t22;

            sol[base] = temp;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_2d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
