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

    // Preload filter values into local variables for better register usage
    TYPE f0 = filter[0], f1 = filter[1], f2 = filter[2];
    TYPE f3 = filter[3], f4 = filter[4], f5 = filter[5];
    TYPE f6 = filter[6], f7 = filter[7], f8 = filter[8];

    #pragma omp parallel for private(c, k1, k2) schedule(static)
    for (r=0; r<row_size-2; r++) {
        for (c=0; c<col_size-2; c++) {
            // Manually unroll the 3x3 stencil computation
            int base_idx = r * col_size + c;
            TYPE temp = f0 * orig[base_idx] + 
                        f1 * orig[base_idx + 1] + 
                        f2 * orig[base_idx + 2] +
                        f3 * orig[base_idx + col_size] + 
                        f4 * orig[base_idx + col_size + 1] + 
                        f5 * orig[base_idx + col_size + 2] +
                        f6 * orig[base_idx + 2*col_size] + 
                        f7 * orig[base_idx + 2*col_size + 1] + 
                        f8 * orig[base_idx + 2*col_size + 2];
            sol[base_idx] = temp;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_2d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
