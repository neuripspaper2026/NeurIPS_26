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

    /* Cache filter coefficients in registers */
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
#ifdef _OPENMP
    #pragma omp parallel for private(c) schedule(static)
#endif
    for (r = 0; r < row_size-2; r++) {
        stencil_label2:for (c = 0; c < col_size-2; c++) {
            TYPE t0, t1, t2, t3, t4, t5, t6, t7, t8;
            TYPE temp;
            int base = r * col_size + c;
            int base1 = base + col_size;
            int base2 = base1 + col_size;

            /* Manually unrolled 3x3 stencil to reduce loop overhead
               and enable better vectorization and instruction scheduling */
            t0 = orig[base];
            t1 = orig[base + 1];
            t2 = orig[base + 2];

            t3 = orig[base1];
            t4 = orig[base1 + 1];
            t5 = orig[base1 + 2];

            t6 = orig[base2];
            t7 = orig[base2 + 1];
            t8 = orig[base2 + 2];

            temp  = f0 * t0;
            temp += f1 * t1;
            temp += f2 * t2;
            temp += f3 * t3;
            temp += f4 * t4;
            temp += f5 * t5;
            temp += f6 * t6;
            temp += f7 * t7;
            temp += f8 * t8;

            sol[base] = temp;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_2d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
