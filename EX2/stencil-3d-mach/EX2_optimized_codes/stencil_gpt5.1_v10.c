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
    TYPE c0 = C[0];
    TYPE c1 = C[1];
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    // Handle boundary conditions by copying from orig to sol
    // Parallelize independent loops over the full 3D domain for better load balance

    // Height boundaries (k and j full, i fixed at 0 and height_size-1)
#ifdef _OPENMP
#pragma omp parallel for collapse(2) private(j,k) schedule(static)
#endif
    for (j = 0; j < col_size; j++) {
        for (k = 0; k < row_size; k++) {
            sol[INDX(row_size, col_size, k, j, 0)] =
                orig[INDX(row_size, col_size, k, j, 0)];
            sol[INDX(row_size, col_size, k, j, height_size - 1)] =
                orig[INDX(row_size, col_size, k, j, height_size - 1)];
        }
    }

    // Column boundaries (k full, j fixed at 0 and col_size-1, inner heights)
#ifdef _OPENMP
#pragma omp parallel for collapse(2) private(i,k) schedule(static)
#endif
    for (i = 1; i < height_size - 1; i++) {
        for (k = 0; k < row_size; k++) {
            sol[INDX(row_size, col_size, k, 0, i)] =
                orig[INDX(row_size, col_size, k, 0, i)];
            sol[INDX(row_size, col_size, k, col_size - 1, i)] =
                orig[INDX(row_size, col_size, k, col_size - 1, i)];
        }
    }

    // Row boundaries (j inner, k fixed at 0 and row_size-1, inner heights)
#ifdef _OPENMP
#pragma omp parallel for collapse(2) private(i,j) schedule(static)
#endif
    for (i = 1; i < height_size - 1; i++) {
        for (j = 1; j < col_size - 1; j++) {
            sol[INDX(row_size, col_size, 0, j, i)] =
                orig[INDX(row_size, col_size, 0, j, i)];
            sol[INDX(row_size, col_size, row_size - 1, j, i)] =
                orig[INDX(row_size, col_size, row_size - 1, j, i)];
        }
    }

    // Stencil computation on interior points
    // Use OpenMP over outer dimensions with collapse for better parallelism.
#ifdef _OPENMP
#pragma omp parallel for collapse(2) private(i,j,k) schedule(static)
#endif
    for (i = 1; i < height_size - 1; i++) {
        for (j = 1; j < col_size - 1; j++) {
#pragma GCC ivdep
            for (k = 1; k < row_size - 1; k++) {
                int idx     = INDX(row_size, col_size, k,     j,     i);
                int idx_ip1 = INDX(row_size, col_size, k,     j,     i + 1);
                int idx_im1 = INDX(row_size, col_size, k,     j,     i - 1);
                int idx_jp1 = INDX(row_size, col_size, k,     j + 1, i);
                int idx_jm1 = INDX(row_size, col_size, k,     j - 1, i);
                int idx_kp1 = INDX(row_size, col_size, k + 1, j,     i);
                int idx_km1 = INDX(row_size, col_size, k - 1, j,     i);

                TYPE center = orig[idx];
                TYPE neigh_sum =
                    orig[idx_ip1] +
                    orig[idx_im1] +
                    orig[idx_jp1] +
                    orig[idx_jm1] +
                    orig[idx_kp1] +
                    orig[idx_km1];

                sol[idx] = center * c0 + neigh_sum * c1;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_3d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
