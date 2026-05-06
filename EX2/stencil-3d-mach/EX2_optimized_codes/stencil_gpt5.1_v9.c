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
    const int rs = row_size;
    const int cs = col_size;
    const int plane_stride = rs * cs;

    // Handle boundary conditions by filling with original values
#ifdef _OPENMP
#pragma omp parallel
    {
#pragma omp for collapse(2) nowait
        for (j = 0; j < cs; j++) {
            for (k = 0; k < rs; k++) {
                const int idx0 = k + rs * (j + cs * 0);
                const int idxh = k + rs * (j + cs * (height_size - 1));
                sol[idx0] = orig[idx0];
                sol[idxh] = orig[idxh];
            }
        }

#pragma omp for collapse(2) nowait
        for (i = 1; i < height_size - 1; i++) {
            for (k = 0; k < rs; k++) {
                const int idx0 = k + rs * (0 + cs * i);
                const int idxc = k + rs * ((cs - 1) + cs * i);
                sol[idx0] = orig[idx0];
                sol[idxc] = orig[idxc];
            }
        }

#pragma omp for collapse(2) nowait
        for (i = 1; i < height_size - 1; i++) {
            for (j = 1; j < cs - 1; j++) {
                const int idx0 = 0 + rs * (j + cs * i);
                const int idxr = (rs - 1) + rs * (j + cs * i);
                sol[idx0] = orig[idx0];
                sol[idxr] = orig[idxr];
            }
        }

        // Stencil computation in interior
#pragma omp for collapse(2)
        for (i = 1; i < height_size - 1; i++) {
            for (j = 1; j < cs - 1; j++) {
#pragma GCC ivdep
                for (k = 1; k < rs - 1; k++) {
                    const int base = k + rs * (j + cs * i);
                    const int ip1  = base + plane_stride;
                    const int im1  = base - plane_stride;
                    const int jp1  = base + rs;
                    const int jm1  = base - rs;
                    const int kp1  = base + 1;
                    const int km1  = base - 1;

                    sum0 = orig[base];
                    sum1 = orig[ip1] + orig[im1] +
                           orig[jp1] + orig[jm1] +
                           orig[kp1] + orig[km1];

                    mul0 = sum0 * C[0];
                    mul1 = sum1 * C[1];
                    sol[base] = mul0 + mul1;
                }
            }
        }
    }
#else
    for (j = 0; j < cs; j++) {
        for (k = 0; k < rs; k++) {
            const int idx0 = k + rs * (j + cs * 0);
            const int idxh = k + rs * (j + cs * (height_size - 1));
            sol[idx0] = orig[idx0];
            sol[idxh] = orig[idxh];
        }
    }

    for (i = 1; i < height_size - 1; i++) {
        for (k = 0; k < rs; k++) {
            const int idx0 = k + rs * (0 + cs * i);
            const int idxc = k + rs * ((cs - 1) + cs * i);
            sol[idx0] = orig[idx0];
            sol[idxc] = orig[idxc];
        }
    }

    for (i = 1; i < height_size - 1; i++) {
        for (j = 1; j < cs - 1; j++) {
            const int idx0 = 0 + rs * (j + cs * i);
            const int idxr = (rs - 1) + rs * (j + cs * i);
            sol[idx0] = orig[idx0];
            sol[idxr] = orig[idxr];
        }
    }

    for (i = 1; i < height_size - 1; i++) {
        for (j = 1; j < cs - 1; j++) {
#pragma GCC ivdep
            for (k = 1; k < rs - 1; k++) {
                const int base = k + rs * (j + cs * i);
                const int ip1  = base + plane_stride;
                const int im1  = base - plane_stride;
                const int jp1  = base + rs;
                const int jm1  = base - rs;
                const int kp1  = base + 1;
                const int km1  = base - 1;

                sum0 = orig[base];
                sum1 = orig[ip1] + orig[im1] +
                       orig[jp1] + orig[jm1] +
                       orig[kp1] + orig[km1];

                mul0 = sum0 * C[0];
                mul1 = sum1 * C[1];
                sol[base] = mul0 + mul1;
            }
        }
    }
#endif

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_3d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
