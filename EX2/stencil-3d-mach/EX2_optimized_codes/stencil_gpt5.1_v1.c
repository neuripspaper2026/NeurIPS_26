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
    const int rs = row_size;
    const int cs = col_size;
    const int hs = height_size;
    const TYPE c0 = C[0];
    const TYPE c1 = C[1];
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    // Handle boundary conditions by filling with original values
#ifdef _OPENMP
#pragma omp parallel
    {
#pragma omp for collapse(2) nowait
#endif
        for (j = 0; j < cs; j++) {
            for (k = 0; k < rs; k++) {
                sol[INDX(rs, cs, k, j, 0)] = orig[INDX(rs, cs, k, j, 0)];
                sol[INDX(rs, cs, k, j, hs - 1)] = orig[INDX(rs, cs, k, j, hs - 1)];
            }
        }
#ifdef _OPENMP
#pragma omp for collapse(2) nowait
#endif
        for (i = 1; i < hs - 1; i++) {
            for (k = 0; k < rs; k++) {
                sol[INDX(rs, cs, k, 0, i)] = orig[INDX(rs, cs, k, 0, i)];
                sol[INDX(rs, cs, k, cs - 1, i)] = orig[INDX(rs, cs, k, cs - 1, i)];
            }
        }
#ifdef _OPENMP
#pragma omp for collapse(2) nowait
#endif
        for (i = 1; i < hs - 1; i++) {
            for (j = 1; j < cs - 1; j++) {
                sol[INDX(rs, cs, 0, j, i)] = orig[INDX(rs, cs, 0, j, i)];
                sol[INDX(rs, cs, rs - 1, j, i)] = orig[INDX(rs, cs, rs - 1, j, i)];
            }
        }

        // Stencil computation
#ifdef _OPENMP
#pragma omp for collapse(3)
#endif
        for (i = 1; i < hs - 1; i++) {
            for (j = 1; j < cs - 1; j++) {
#pragma GCC ivdep
                for (k = 1; k < rs - 1; k++) {
                    const int idx    = INDX(rs, cs, k,     j,     i);
                    const int idx_im = INDX(rs, cs, k,     j,     i - 1);
                    const int idx_ip = INDX(rs, cs, k,     j,     i + 1);
                    const int idx_jm = INDX(rs, cs, k,     j - 1, i);
                    const int idx_jp = INDX(rs, cs, k,     j + 1, i);
                    const int idx_km = INDX(rs, cs, k - 1, j,     i);
                    const int idx_kp = INDX(rs, cs, k + 1, j,     i);

                    TYPE center = orig[idx];
                    TYPE neigh_sum =
                        orig[idx_ip] +
                        orig[idx_im] +
                        orig[idx_jp] +
                        orig[idx_jm] +
                        orig[idx_kp] +
                        orig[idx_km];

                    sol[idx] = center * c0 + neigh_sum * c1;
                }
            }
        }
#ifdef _OPENMP
    }
#endif

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_3d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
