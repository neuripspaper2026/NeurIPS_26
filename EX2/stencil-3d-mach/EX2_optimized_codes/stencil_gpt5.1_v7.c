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
    height_bound_col : for (j = 0; j < col_size; j++) {
        height_bound_row : for (k = 0; k < row_size; k++) {
            const int idx0 = INDX(row_size, col_size, k, j, 0);
            const int idxh = INDX(row_size, col_size, k, j, height_size - 1);
            sol[idx0] = orig[idx0];
            sol[idxh] = orig[idxh];
        }
    }
    col_bound_height : for (i = 1; i < height_size - 1; i++) {
        col_bound_row : for (k = 0; k < row_size; k++) {
            const int idx0 = INDX(row_size, col_size, k, 0, i);
            const int idxc = INDX(row_size, col_size, k, col_size - 1, i);
            sol[idx0] = orig[idx0];
            sol[idxc] = orig[idxc];
        }
    }
    row_bound_height : for (i = 1; i < height_size - 1; i++) {
        row_bound_col : for (j = 1; j < col_size - 1; j++) {
            const int idx0 = INDX(row_size, col_size, 0, j, i);
            const int idxr = INDX(row_size, col_size, row_size - 1, j, i);
            sol[idx0] = orig[idx0];
            sol[idxr] = orig[idxr];
        }
    }

    const TYPE c0 = C[0];
    const TYPE c1 = C[1];
    const int rs = row_size;
    const int cs = col_size;

    // Stencil computation
#ifdef _OPENMP
#pragma omp parallel for collapse(3) private(i, j, k, sum0, sum1, mul0, mul1) schedule(static)
#endif
    for (i = 1; i < height_size - 1; i++) {
        for (j = 1; j < col_size - 1; j++) {
            for (k = 1; k < row_size - 1; k++) {
                const int idx    = INDX(rs, cs, k,     j,     i);
                const int idx_ip = INDX(rs, cs, k,     j,     i + 1);
                const int idx_im = INDX(rs, cs, k,     j,     i - 1);
                const int idx_jp = INDX(rs, cs, k,     j + 1, i);
                const int idx_jm = INDX(rs, cs, k,     j - 1, i);
                const int idx_kp = INDX(rs, cs, k + 1, j,     i);
                const int idx_km = INDX(rs, cs, k - 1, j,     i);

                sum0 = orig[idx];
                sum1 = orig[idx_ip] +
                       orig[idx_im] +
                       orig[idx_jp] +
                       orig[idx_jm] +
                       orig[idx_kp] +
                       orig[idx_km];
                mul0 = sum0 * c0;
                mul1 = sum1 * c1;
                sol[idx] = mul0 + mul1;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_3d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
