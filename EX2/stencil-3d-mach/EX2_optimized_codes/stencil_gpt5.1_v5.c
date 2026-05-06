#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../stencil.h"

static double stencil_3d_kernel_time_acc = 0.0;

void reset_stencil_3d_kernel_time(void) { stencil_3d_kernel_time_acc = 0.0; }
double get_stencil_3d_kernel_time(void) { return stencil_3d_kernel_time_acc; }

void stencil3d(TYPE C[2], TYPE orig[SIZE], TYPE sol[SIZE]) {
    TYPE c0 = C[0];
    TYPE c1 = C[1];

    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    // Handle boundary conditions by filling with original values
    // Parallelize across independent loops; each iteration writes unique elements
#ifdef _OPENMP
#pragma omp parallel
    {
#pragma omp for collapse(2) nowait
#endif
    for (int j = 0; j < col_size; j++) {
        for (int k = 0; k < row_size; k++) {
            sol[INDX(row_size, col_size, k, j, 0)] =
                orig[INDX(row_size, col_size, k, j, 0)];
            sol[INDX(row_size, col_size, k, j, height_size - 1)] =
                orig[INDX(row_size, col_size, k, j, height_size - 1)];
        }
    }

#ifdef _OPENMP
#pragma omp for collapse(2) nowait
#endif
    for (int i = 1; i < height_size - 1; i++) {
        for (int k = 0; k < row_size; k++) {
            sol[INDX(row_size, col_size, k, 0, i)] =
                orig[INDX(row_size, col_size, k, 0, i)];
            sol[INDX(row_size, col_size, k, col_size - 1, i)] =
                orig[INDX(row_size, col_size, k, col_size - 1, i)];
        }
    }

#ifdef _OPENMP
#pragma omp for collapse(2) nowait
#endif
    for (int i = 1; i < height_size - 1; i++) {
        for (int j = 1; j < col_size - 1; j++) {
            sol[INDX(row_size, col_size, 0, j, i)] =
                orig[INDX(row_size, col_size, 0, j, i)];
            sol[INDX(row_size, col_size, row_size - 1, j, i)] =
                orig[INDX(row_size, col_size, row_size - 1, j, i)];
        }
    }

    // Stencil computation
#ifdef _OPENMP
#pragma omp for collapse(3) nowait
#endif
    for (int i = 1; i < height_size - 1; i++) {
        for (int j = 1; j < col_size - 1; j++) {
            for (int k = 1; k < row_size - 1; k++) {

                // Compute base index once and use neighbor offsets
                const int base = INDX(row_size, col_size, k, j, i);

                TYPE center = orig[base];
                TYPE sum1 =
                    orig[base + row_size * col_size] +         // (i+1)
                    orig[base - row_size * col_size] +         // (i-1)
                    orig[base + row_size] +                    // (j+1)
                    orig[base - row_size] +                    // (j-1)
                    orig[base + 1] +                           // (k+1)
                    orig[base - 1];                            // (k-1)

                sol[base] = center * c0 + sum1 * c1;
            }
        }
    }
#ifdef _OPENMP
    } // end parallel
#endif

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_3d_kernel_time_acc +=
        (kernel_end.tv_sec - kernel_start.tv_sec) +
        (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
