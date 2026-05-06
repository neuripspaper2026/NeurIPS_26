#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../stencil.h"

static double stencil_3d_kernel_time_acc = 0.0;

void reset_stencil_3d_kernel_time(void) { stencil_3d_kernel_time_acc = 0.0; }
double get_stencil_3d_kernel_time(void) { return stencil_3d_kernel_time_acc; }

void stencil3d(TYPE C[2], TYPE orig[SIZE], TYPE sol[SIZE]) {
    int i, j, k;
    TYPE sum0, sum1, mul0, mul1;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    // Handle boundary conditions by filling with original values
#ifdef _OPENMP
#pragma omp parallel
    {
#pragma omp for collapse(2) nowait
#endif
        for (j = 0; j < col_size; j++) {
            for (k = 0; k < row_size; k++) {
                sol[INDX(row_size, col_size, k, j, 0)] =
                    orig[INDX(row_size, col_size, k, j, 0)];
                sol[INDX(row_size, col_size, k, j, height_size - 1)] =
                    orig[INDX(row_size, col_size, k, j, height_size - 1)];
            }
        }

#ifdef _OPENMP
#pragma omp for collapse(2) nowait
#endif
        for (i = 1; i < height_size - 1; i++) {
            for (k = 0; k < row_size; k++) {
                sol[INDX(row_size, col_size, k, 0, i)] =
                    orig[INDX(row_size, col_size, k, 0, i)];
                sol[INDX(row_size, col_size, k, col_size - 1, i)] =
                    orig[INDX(row_size, col_size, k, col_size - 1, i)];
            }
        }

#ifdef _OPENMP
#pragma omp for collapse(2) nowait
#endif
        for (i = 1; i < height_size - 1; i++) {
            for (j = 1; j < col_size - 1; j++) {
                sol[INDX(row_size, col_size, 0, j, i)] =
                    orig[INDX(row_size, col_size, 0, j, i)];
                sol[INDX(row_size, col_size, row_size - 1, j, i)] =
                    orig[INDX(row_size, col_size, row_size - 1, j, i)];
            }
        }

        // Stencil computation
#ifdef _OPENMP
#pragma omp for collapse(3)
#endif
        for (i = 1; i < height_size - 1; i++) {
            for (j = 1; j < col_size - 1; j++) {
#pragma GCC ivdep
                for (k = 1; k < row_size - 1; k++) {
                    const int idx    = INDX(row_size, col_size, k,     j,     i);
                    const int idx_ip = INDX(row_size, col_size, k,     j,     i + 1);
                    const int idx_im = INDX(row_size, col_size, k,     j,     i - 1);
                    const int idx_jp = INDX(row_size, col_size, k,     j + 1, i);
                    const int idx_jm = INDX(row_size, col_size, k,     j - 1, i);
                    const int idx_kp = INDX(row_size, col_size, k + 1, j,     i);
                    const int idx_km = INDX(row_size, col_size, k - 1, j,     i);

                    sum0 = orig[idx];
                    sum1 = orig[idx_ip] +
                           orig[idx_im] +
                           orig[idx_jp] +
                           orig[idx_jm] +
                           orig[idx_kp] +
                           orig[idx_km];
                    mul0 = sum0 * C[0];
                    mul1 = sum1 * C[1];
                    sol[idx] = mul0 + mul1;
                }
            }
        }
#ifdef _OPENMP
    }
#endif

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_3d_kernel_time_acc +=
        (kernel_end.tv_sec - kernel_start.tv_sec) +
        (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
