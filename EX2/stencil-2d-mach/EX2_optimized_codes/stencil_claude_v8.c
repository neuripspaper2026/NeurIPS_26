#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../stencil.h"

static double stencil_2d_kernel_time_acc = 0.0;

void reset_stencil_2d_kernel_time(void) { stencil_2d_kernel_time_acc = 0.0; }
double get_stencil_2d_kernel_time(void) { return stencil_2d_kernel_time_acc; }

void stencil (TYPE orig[row_size * col_size], TYPE sol[row_size * col_size], TYPE filter[f_size]){
    int r, c, k1, k2;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    #pragma omp parallel
    {
        TYPE f[9];
        #pragma omp for nowait
        for (int i = 0; i < 9; i++) {
            f[i] = filter[i];
        }

        #pragma omp for schedule(static) nowait
        stencil_label1:for (r=0; r<row_size-2; r++) {
            for (c=0; c<col_size-2; c++) {
                TYPE temp = orig[r*col_size + c] * f[0] + 
                           orig[r*col_size + c+1] * f[1] + 
                           orig[r*col_size + c+2] * f[2] +
                           orig[(r+1)*col_size + c] * f[3] + 
                           orig[(r+1)*col_size + c+1] * f[4] + 
                           orig[(r+1)*col_size + c+2] * f[5] +
                           orig[(r+2)*col_size + c] * f[6] + 
                           orig[(r+2)*col_size + c+1] * f[7] + 
                           orig[(r+2)*col_size + c+2] * f[8];
                sol[r*col_size + c] = temp;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_2d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
