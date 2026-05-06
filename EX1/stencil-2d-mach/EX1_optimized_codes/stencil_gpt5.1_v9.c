#include <time.h>
#include "../stencil.h"

static double stencil_2d_kernel_time_acc = 0.0;

void reset_stencil_2d_kernel_time(void) { stencil_2d_kernel_time_acc = 0.0; }
double get_stencil_2d_kernel_time(void) { return stencil_2d_kernel_time_acc; }

void stencil (TYPE orig[row_size * col_size], TYPE sol[row_size * col_size], TYPE filter[f_size]){
    int r, c;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    for (r = 0; r < row_size - 2; r++) {
        TYPE *orig_row0 = &orig[r * col_size];
        TYPE *orig_row1 = orig_row0 + col_size;
        TYPE *orig_row2 = orig_row1 + col_size;
        TYPE *sol_row   = &sol[r * col_size];

        for (c = 0; c < col_size - 2; c++) {
            TYPE v00 = orig_row0[c];
            TYPE v01 = orig_row0[c + 1];
            TYPE v02 = orig_row0[c + 2];

            TYPE v10 = orig_row1[c];
            TYPE v11 = orig_row1[c + 1];
            TYPE v12 = orig_row1[c + 2];

            TYPE v20 = orig_row2[c];
            TYPE v21 = orig_row2[c + 1];
            TYPE v22 = orig_row2[c + 2];

            TYPE t0 = filter[0] * v00 + filter[1] * v01 + filter[2] * v02;
            TYPE t1 = filter[3] * v10 + filter[4] * v11 + filter[5] * v12;
            TYPE t2 = filter[6] * v20 + filter[7] * v21 + filter[8] * v22;

            sol_row[c] = t0 + t1 + t2;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_2d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
