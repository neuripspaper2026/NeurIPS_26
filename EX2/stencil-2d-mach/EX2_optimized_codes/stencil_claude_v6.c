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
    TYPE temp, mul;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    #pragma omp parallel for private(c, k1, k2, temp, mul) schedule(static)
    stencil_label1:for (r=0; r<row_size-2; r++) {
        stencil_label2:for (c=0; c<col_size-2; c++) {
            temp = (TYPE)0;
            
            // Unroll inner loops manually for better performance
            temp += filter[0] * orig[r*col_size + c];
            temp += filter[1] * orig[r*col_size + c+1];
            temp += filter[2] * orig[r*col_size + c+2];
            
            temp += filter[3] * orig[(r+1)*col_size + c];
            temp += filter[4] * orig[(r+1)*col_size + c+1];
            temp += filter[5] * orig[(r+1)*col_size + c+2];
            
            temp += filter[6] * orig[(r+2)*col_size + c];
            temp += filter[7] * orig[(r+2)*col_size + c+1];
            temp += filter[8] * orig[(r+2)*col_size + c+2];
            
            sol[(r*col_size) + c] = temp;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_2d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
