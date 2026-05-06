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

    /* Cache filter values in registers */
    f0 = filter[0];
    f1 = filter[1];
    f2 = filter[2];
    f3 = filter[3];
    f4 = filter[4];
    f5 = filter[5];
    f6 = filter[6];
    f7 = filter[7];
    f8 = filter[8];

    stencil_label1:
    #pragma omp parallel for private(c) schedule(static)
    for (r = 0; r < row_size - 2; r++) {
        stencil_label2:
        for (c = 0; c < col_size - 2; c++) {
            int idx_rc = r * col_size + c;
            int idx_r1 = idx_rc + col_size;
            int idx_r2 = idx_r1 + col_size;

            TYPE v0 = orig[idx_rc];
            TYPE v1 = orig[idx_rc + 1];
            TYPE v2 = orig[idx_rc + 2];

            TYPE v3 = orig[idx_r1];
            TYPE v4 = orig[idx_r1 + 1];
            TYPE v5 = orig[idx_r1 + 2];

            TYPE v6 = orig[idx_r2];
            TYPE v7 = orig[idx_r2 + 1];
            TYPE v8 = orig[idx_r2 + 2];

            TYPE temp =  v0 * f0 +
                         v1 * f1 +
                         v2 * f2 +
                         v3 * f3 +
                         v4 * f4 +
                         v5 * f5 +
                         v6 * f6 +
                         v7 * f7 +
                         v8 * f8;

            sol[idx_rc] = temp;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_2d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
