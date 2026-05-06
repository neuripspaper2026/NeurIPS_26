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

    /* Cache filter values to avoid repeated memory loads */
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
        int base_r0 = r * col_size;
        int base_r1 = (r + 1) * col_size;
        int base_r2 = (r + 2) * col_size;

        stencil_label2:
        #pragma GCC ivdep
        #pragma clang loop vectorize(enable)
        for (c = 0; c < col_size - 2; c++) {
            TYPE t0, t1, t2, t3, t4, t5, t6, t7, t8;
            TYPE temp;

            t0 = orig[base_r0 + c];
            t1 = orig[base_r0 + c + 1];
            t2 = orig[base_r0 + c + 2];

            t3 = orig[base_r1 + c];
            t4 = orig[base_r1 + c + 1];
            t5 = orig[base_r1 + c + 2];

            t6 = orig[base_r2 + c];
            t7 = orig[base_r2 + c + 1];
            t8 = orig[base_r2 + c + 2];

            /* Fully unrolled 3x3 stencil computation */
            temp  = f0 * t0;
            temp += f1 * t1;
            temp += f2 * t2;
            temp += f3 * t3;
            temp += f4 * t4;
            temp += f5 * t5;
            temp += f6 * t6;
            temp += f7 * t7;
            temp += f8 * t8;

            sol[base_r0 + c] = temp;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_2d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
