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

    // Handle boundary conditions by filling with original values
    height_bound_col : for(j = 0; j < col_size; j++) {
        height_bound_row : for(k = 0; k < row_size; k++) {
            sol[INDX(row_size, col_size, k, j, 0)] =
                orig[INDX(row_size, col_size, k, j, 0)];
            sol[INDX(row_size, col_size, k, j, height_size - 1)] =
                orig[INDX(row_size, col_size, k, j, height_size - 1)];
        }
    }
    col_bound_height : for(i = 1; i < height_size - 1; i++) {
        col_bound_row : for(k = 0; k < row_size; k++) {
            sol[INDX(row_size, col_size, k, 0, i)] =
                orig[INDX(row_size, col_size, k, 0, i)];
            sol[INDX(row_size, col_size, k, col_size - 1, i)] =
                orig[INDX(row_size, col_size, k, col_size - 1, i)];
        }
    }
    row_bound_height : for(i = 1; i < height_size - 1; i++) {
        row_bound_col : for(j = 1; j < col_size - 1; j++) {
            sol[INDX(row_size, col_size, 0, j, i)] =
                orig[INDX(row_size, col_size, 0, j, i)];
            sol[INDX(row_size, col_size, row_size - 1, j, i)] =
                orig[INDX(row_size, col_size, row_size - 1, j, i)];
        }
    }

    // Stencil computation
    loop_height :
#ifdef _OPENMP
    #pragma omp parallel for private(j,k) firstprivate(c0,c1) schedule(static)
#endif
    for(i = 1; i < height_size - 1; i++) {
        loop_col : for(j = 1; j < col_size - 1; j++) {
            loop_row : for(k = 1; k < row_size - 1; k++) {
                const int idx    = INDX(row_size, col_size, k,     j,     i    );
                const int idx_ip = INDX(row_size, col_size, k,     j,     i + 1);
                const int idx_im = INDX(row_size, col_size, k,     j,     i - 1);
                const int idx_jp = INDX(row_size, col_size, k,     j + 1, i    );
                const int idx_jm = INDX(row_size, col_size, k,     j - 1, i    );
                const int idx_kp = INDX(row_size, col_size, k + 1, j,     i    );
                const int idx_km = INDX(row_size, col_size, k - 1, j,     i    );

                TYPE center = orig[idx];
                TYPE sum1   = orig[idx_ip] +
                              orig[idx_im] +
                              orig[idx_jp] +
                              orig[idx_jm] +
                              orig[idx_kp] +
                              orig[idx_km];

                sol[idx] = center * c0 + sum1 * c1;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_3d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
