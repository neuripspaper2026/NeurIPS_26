#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../stencil.h"

static double stencil_3d_kernel_time_acc = 0.0;

void reset_stencil_3d_kernel_time(void) { stencil_3d_kernel_time_acc = 0.0; }
double get_stencil_3d_kernel_time(void) { return stencil_3d_kernel_time_acc; }

void stencil3d(TYPE C[2], TYPE orig[SIZE], TYPE sol[SIZE]) {
    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    // Handle boundary conditions by filling with original values
    #pragma omp parallel
    {
        int j, k, i;
        
        #pragma omp for nowait
        for(j = 0; j < col_size; j++) {
            for(k = 0; k < row_size; k++) {
                sol[INDX(row_size, col_size, k, j, 0)] = orig[INDX(row_size, col_size, k, j, 0)];
                sol[INDX(row_size, col_size, k, j, height_size-1)] = orig[INDX(row_size, col_size, k, j, height_size-1)];
            }
        }

        #pragma omp for nowait
        for(i = 1; i < height_size-1; i++) {
            for(k = 0; k < row_size; k++) {
                sol[INDX(row_size, col_size, k, 0, i)] = orig[INDX(row_size, col_size, k, 0, i)];
                sol[INDX(row_size, col_size, k, col_size-1, i)] = orig[INDX(row_size, col_size, k, col_size-1, i)];
            }
        }

        #pragma omp for nowait
        for(i = 1; i < height_size-1; i++) {
            for(j = 1; j < col_size-1; j++) {
                sol[INDX(row_size, col_size, 0, j, i)] = orig[INDX(row_size, col_size, 0, j, i)];
                sol[INDX(row_size, col_size, row_size-1, j, i)] = orig[INDX(row_size, col_size, row_size-1, j, i)];
            }
        }
    }

    // Stencil computation
    const TYPE c0 = C[0];
    const TYPE c1 = C[1];
    
    #pragma omp parallel for collapse(3) schedule(static)
    for(int i = 1; i < height_size - 1; i++){
        for(int j = 1; j < col_size - 1; j++){
            for(int k = 1; k < row_size - 1; k++){
                const TYPE sum0 = orig[INDX(row_size, col_size, k, j, i)];
                const TYPE sum1 = orig[INDX(row_size, col_size, k, j, i + 1)] +
                                 orig[INDX(row_size, col_size, k, j, i - 1)] +
                                 orig[INDX(row_size, col_size, k, j + 1, i)] +
                                 orig[INDX(row_size, col_size, k, j - 1, i)] +
                                 orig[INDX(row_size, col_size, k + 1, j, i)] +
                                 orig[INDX(row_size, col_size, k - 1, j, i)];
                sol[INDX(row_size, col_size, k, j, i)] = sum0 * c0 + sum1 * c1;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_3d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
