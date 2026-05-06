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

    // Precompute strides for faster indexing
    const int row_stride    = 1;
    const int col_stride    = row_size;
    const int height_stride = row_size * col_size;

    // Handle boundary conditions by filling with original values
#ifdef _OPENMP
#pragma omp parallel for private(j,k) collapse(2) schedule(static)
#endif
    for (j = 0; j < col_size; j++) {
        for (k = 0; k < row_size; k++) {
            int idx0 = k * row_stride + j * col_stride + 0 * height_stride;
            int idxH = k * row_stride + j * col_stride + (height_size - 1) * height_stride;
            sol[idx0] = orig[idx0];
            sol[idxH] = orig[idxH];
        }
    }

#ifdef _OPENMP
#pragma omp parallel for private(k) collapse(2) schedule(static)
#endif
    for (i = 1; i < height_size - 1; i++) {
        for (k = 0; k < row_size; k++) {
            int idxC0 = k * row_stride + 0 * col_stride + i * height_stride;
            int idxC1 = k * row_stride + (col_size - 1) * col_stride + i * height_stride;
            sol[idxC0] = orig[idxC0];
            sol[idxC1] = orig[idxC1];
        }
    }

#ifdef _OPENMP
#pragma omp parallel for private(j) collapse(2) schedule(static)
#endif
    for (i = 1; i < height_size - 1; i++) {
        for (j = 1; j < col_size - 1; j++) {
            int idxR0 = 0 * row_stride + j * col_stride + i * height_stride;
            int idxR1 = (row_size - 1) * row_stride + j * col_stride + i * height_stride;
            sol[idxR0] = orig[idxR0];
            sol[idxR1] = orig[idxR1];
        }
    }

    // Stencil computation
#ifdef _OPENMP
#pragma omp parallel for private(j,k,sum0,sum1,mul0,mul1) collapse(2) schedule(static)
#endif
    for (i = 1; i < height_size - 1; i++) {
        for (j = 1; j < col_size - 1; j++) {
            int base_ij   = j * col_stride + i * height_stride;
            int base_ip1  = j * col_stride + (i + 1) * height_stride;
            int base_im1  = j * col_stride + (i - 1) * height_stride;
            int base_jp1  = (j + 1) * col_stride + i * height_stride;
            int base_jm1  = (j - 1) * col_stride + i * height_stride;

            for (k = 1; k < row_size - 1; k++) {
                int center = base_ij + k * row_stride;

                sum0 = orig[center];
                sum1 = orig[base_ip1 + k * row_stride] +
                       orig[base_im1 + k * row_stride] +
                       orig[base_jp1 + k * row_stride] +
                       orig[base_jm1 + k * row_stride] +
                       orig[base_ij + (k + 1) * row_stride] +
                       orig[base_ij + (k - 1) * row_stride];

                mul0 = sum0 * C[0];
                mul1 = sum1 * C[1];
                sol[center] = mul0 + mul1;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_3d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
