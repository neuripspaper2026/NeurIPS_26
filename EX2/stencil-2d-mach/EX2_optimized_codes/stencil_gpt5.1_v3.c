#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../stencil.h"

static double stencil_2d_kernel_time_acc = 0.0;

void reset_stencil_2d_kernel_time(void) { stencil_2d_kernel_time_acc = 0.0; }
double get_stencil_2d_kernel_time(void) { return stencil_2d_kernel_time_acc; }

void stencil (TYPE orig[row_size * col_size], TYPE sol[row_size * col_size], TYPE filter[f_size]){
    int r, c, k;
    TYPE temp;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    /* Preload filter values into local scalars to help the compiler */
    TYPE f0 = filter[0];
    TYPE f1 = filter[1];
    TYPE f2 = filter[2];
    TYPE f3 = filter[3];
    TYPE f4 = filter[4];
    TYPE f5 = filter[5];
    TYPE f6 = filter[6];
    TYPE f7 = filter[7];
    TYPE f8 = filter[8];

    stencil_label1:
#ifdef _OPENMP
#pragma omp parallel for private(c, temp) schedule(static)
#endif
    for (r = 0; r < row_size - 2; r++) {
        stencil_label2:for (c = 0; c < col_size - 2; c++) {
            /* Fully unroll 3x3 stencil computation */
            TYPE v0 = orig[(r    ) * col_size + (c    )];
            TYPE v1 = orig[(r    ) * col_size + (c + 1)];
            TYPE v2 = orig[(r    ) * col_size + (c + 2)];
            TYPE v3 = orig[(r + 1) * col_size + (c    )];
            TYPE v4 = orig[(r + 1) * col_size + (c + 1)];
            TYPE v5 = orig[(r + 1) * col_size + (c + 2)];
            TYPE v6 = orig[(r + 2) * col_size + (c    )];
            TYPE v7 = orig[(r + 2) * col_size + (c + 1)];
            TYPE v8 = orig[(r + 2) * col_size + (c + 2)];

            temp  = f0 * v0;
            temp += f1 * v1;
            temp += f2 * v2;
            temp += f3 * v3;
            temp += f4 * v4;
            temp += f5 * v5;
            temp += f6 * v6;
            temp += f7 * v7;
            temp += f8 * v8;

            sol[(r * col_size) + c] = temp;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_2d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
