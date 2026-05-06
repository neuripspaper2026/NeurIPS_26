#include <time.h>
#include "../stencil.h"

static double stencil_3d_kernel_time_acc = 0.0;

void reset_stencil_3d_kernel_time(void) { stencil_3d_kernel_time_acc = 0.0; }
double get_stencil_3d_kernel_time(void) { return stencil_3d_kernel_time_acc; }

void stencil3d(TYPE C[2], TYPE orig[SIZE], TYPE sol[SIZE]) {
    int i, j, k;
    TYPE sum0, sum1, mul0, mul1;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    // Handle boundary conditions by copying boundary directly
    for (i = 0; i < height_size; ++i) {
        for (j = 0; j < col_size; ++j) {
            for (k = 0; k < row_size; ++k) {
                int idx = INDX(row_size, col_size, k, j, i);
                if (i == 0 || i == height_size - 1 ||
                    j == 0 || j == col_size - 1 ||
                    k == 0 || k == row_size - 1) {
                    sol[idx] = orig[idx];
                }
            }
        }
    }

    // Precompute strides for indexing
    const int row_stride = 1;
    const int col_stride = row_size;
    const int height_stride = row_size * col_size;

    // Stencil computation on interior points only
    for (i = 1; i < height_size - 1; ++i) {
        int base_i = i * height_stride;
        int base_ip = base_i + height_stride;
        int base_im = base_i - height_stride;
        for (j = 1; j < col_size - 1; ++j) {
            int base_ij = base_i + j * col_stride;
            int base_ijp = base_ij + col_stride;
            int base_ijm = base_ij - col_stride;
            for (k = 1; k < row_size - 1; ++k) {
                int center = base_ij + k * row_stride;

                sum0 = orig[center];

                sum1 = orig[base_ip + j * col_stride + k] +  // i+1, j, k
                       orig[base_im + j * col_stride + k] +  // i-1, j, k
                       orig[base_ijp + k] +                  // i, j+1, k
                       orig[base_ijm + k] +                  // i, j-1, k
                       orig[center + 1] +                    // i, j, k+1
                       orig[center - 1];                     // i, j, k-1

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
