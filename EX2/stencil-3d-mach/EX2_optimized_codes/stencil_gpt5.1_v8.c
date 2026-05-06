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

    const int rs = row_size;
    const int cs = col_size;
    const int hs = height_size;
    const int plane_size = rs * cs;

    // Handle boundary conditions by filling with original values
#ifdef _OPENMP
#pragma omp parallel
    {
#pragma omp for collapse(2) nowait
#endif
        for (int j = 0; j < cs; j++) {
            for (int k = 0; k < rs; k++) {
                int base0 = INDX(rs, cs, k, j, 0);
                int baseh = INDX(rs, cs, k, j, hs - 1);
                sol[base0] = orig[base0];
                sol[baseh] = orig[baseh];
            }
        }

#ifdef _OPENMP
#pragma omp for collapse(2) nowait
#endif
        for (int i = 1; i < hs - 1; i++) {
            for (int k = 0; k < rs; k++) {
                int base0 = INDX(rs, cs, k, 0, i);
                int basec = INDX(rs, cs, k, cs - 1, i);
                sol[base0] = orig[base0];
                sol[basec] = orig[basec];
            }
        }

#ifdef _OPENMP
#pragma omp for collapse(2) nowait
#endif
        for (int i = 1; i < hs - 1; i++) {
            for (int j = 1; j < cs - 1; j++) {
                int base0 = INDX(rs, cs, 0, j, i);
                int baser = INDX(rs, cs, rs - 1, j, i);
                sol[base0] = orig[base0];
                sol[baser] = orig[baser];
            }
        }

        // Stencil computation
#ifdef _OPENMP
#pragma omp for collapse(3)
#endif
        for (int i = 1; i < hs - 1; i++) {
            for (int j = 1; j < cs - 1; j++) {
                for (int k = 1; k < rs - 1; k++) {
                    int idx     = k + rs * (j + cs * i);
                    int idx_im1 = idx - plane_size;
                    int idx_ip1 = idx + plane_size;
                    int idx_jm1 = idx - rs;
                    int idx_jp1 = idx + rs;
                    int idx_km1 = idx - 1;
                    int idx_kp1 = idx + 1;

                    TYPE center = orig[idx];
                    TYPE sum1 = orig[idx_ip1] + orig[idx_im1] +
                                orig[idx_jp1] + orig[idx_jm1] +
                                orig[idx_kp1] + orig[idx_km1];

                    sol[idx] = center * C[0] + sum1 * C[1];
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
