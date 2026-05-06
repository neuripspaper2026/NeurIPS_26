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

    /* Preload filter values for better cache/register use */
    f0 = filter[0]; f1 = filter[1]; f2 = filter[2];
    f3 = filter[3]; f4 = filter[4]; f5 = filter[5];
    f6 = filter[6]; f7 = filter[7]; f8 = filter[8];

    /* Parallelize outer loop; r and c are private by default, sol and orig shared */
    stencil_label1:
    #pragma omp parallel for private(c) schedule(static)
    for (r = 0; r < row_size - 2; r++) {
        stencil_label2:
        for (c = 0; c < col_size - 2; c++) {
            TYPE t0, t1, t2;
            TYPE o0, o1, o2, o3, o4, o5, o6, o7, o8;
            int base = r * col_size + c;

            /* Manually unrolled 3x3 stencil to remove inner loops and index recomputation */
            o0 = orig[base];
            o1 = orig[base + 1];
            o2 = orig[base + 2];

            o3 = orig[base + col_size];
            o4 = orig[base + col_size + 1];
            o5 = orig[base + col_size + 2];

            o6 = orig[base + (col_size << 1)];
            o7 = orig[base + (col_size << 1) + 1];
            o8 = orig[base + (col_size << 1) + 2];

            t0 = f0 * o0 + f1 * o1 + f2 * o2;
            t1 = f3 * o3 + f4 * o4 + f5 * o5;
            t2 = f6 * o6 + f7 * o7 + f8 * o8;

            sol[base] = t0 + t1 + t2;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_2d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
