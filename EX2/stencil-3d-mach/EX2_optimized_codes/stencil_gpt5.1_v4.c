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
    const int row_stride  = 1;
    const int col_stride  = row_size;
    const int plane_stride = row_size * col_size;

    // Handle boundary conditions by filling with original values
    // Parallelize all boundary loops where safe (no dependencies).
#ifdef _OPENMP
#pragma omp parallel
    {
#pragma omp for collapse(2) nowait
        for (j = 0; j < col_size; j++) {
            for (k = 0; k < row_size; k++) {
                int base0 = k * row_stride + j * col_stride;
                int idx0_front = base0;                       // i = 0
                int idx0_back  = base0 + (height_size - 1) * plane_stride;
                sol[idx0_front] = orig[idx0_front];
                sol[idx0_back]  = orig[idx0_back];
            }
        }

#pragma omp for collapse(2) nowait
        for (i = 1; i < height_size - 1; i++) {
            for (k = 0; k < row_size; k++) {
                int base = k * row_stride + i * plane_stride;
                int idx_left  = base;                        // j = 0
                int idx_right = base + (col_size - 1) * col_stride;
                sol[idx_left]  = orig[idx_left];
                sol[idx_right] = orig[idx_right];
            }
        }

#pragma omp for collapse(2)
        for (i = 1; i < height_size - 1; i++) {
            for (j = 1; j < col_size - 1; j++) {
                int base = j * col_stride + i * plane_stride;
                int idx_front = base;                        // k = 0
                int idx_back  = base + (row_size - 1) * row_stride;
                sol[idx_front] = orig[idx_front];
                sol[idx_back]  = orig[idx_back];
            }
        }

#pragma omp for collapse(3)
        for (i = 1; i < height_size - 1; i++) {
            for (j = 1; j < col_size - 1; j++) {
                for (k = 1; k < row_size - 1; k++) {
                    int idx = k * row_stride + j * col_stride + i * plane_stride;

                    TYPE center = orig[idx];

                    TYPE up    = orig[idx + plane_stride];
                    TYPE down  = orig[idx - plane_stride];
                    TYPE north = orig[idx + col_stride];
                    TYPE south = orig[idx - col_stride];
                    TYPE east  = orig[idx + row_stride];
                    TYPE west  = orig[idx - row_stride];

                    sum0 = center;
                    sum1 = up + down + north + south + east + west;

                    mul0 = sum0 * C[0];
                    mul1 = sum1 * C[1];

                    sol[idx] = mul0 + mul1;
                }
            }
        }
    }
#else
    // Serial version with optimized indexing

    // Height boundaries (i = 0 and i = height_size - 1)
    for (j = 0; j < col_size; j++) {
        for (k = 0; k < row_size; k++) {
            int base0 = k * row_stride + j * col_stride;
            int idx0_front = base0;                       // i = 0
            int idx0_back  = base0 + (height_size - 1) * plane_stride;
            sol[idx0_front] = orig[idx0_front];
            sol[idx0_back]  = orig[idx0_back];
        }
    }

    // Column boundaries (j = 0 and j = col_size - 1)
    for (i = 1; i < height_size - 1; i++) {
        for (k = 0; k < row_size; k++) {
            int base = k * row_stride + i * plane_stride;
            int idx_left  = base;                        // j = 0
            int idx_right = base + (col_size - 1) * col_stride;
            sol[idx_left]  = orig[idx_left];
            sol[idx_right] = orig[idx_right];
        }
    }

    // Row boundaries (k = 0 and k = row_size - 1)
    for (i = 1; i < height_size - 1; i++) {
        for (j = 1; j < col_size - 1; j++) {
            int base = j * col_stride + i * plane_stride;
            int idx_front = base;                        // k = 0
            int idx_back  = base + (row_size - 1) * row_stride;
            sol[idx_front] = orig[idx_front];
            sol[idx_back]  = orig[idx_back];
        }
    }

    // Stencil computation
    for (i = 1; i < height_size - 1; i++) {
        for (j = 1; j < col_size - 1; j++) {
            for (k = 1; k < row_size - 1; k++) {
                int idx = k * row_stride + j * col_stride + i * plane_stride;

                TYPE center = orig[idx];

                TYPE up    = orig[idx + plane_stride];
                TYPE down  = orig[idx - plane_stride];
                TYPE north = orig[idx + col_stride];
                TYPE south = orig[idx - col_stride];
                TYPE east  = orig[idx + row_stride];
                TYPE west  = orig[idx - row_stride];

                sum0 = center;
                sum1 = up + down + north + south + east + west;

                mul0 = sum0 * C[0];
                mul1 = sum1 * C[1];

                sol[idx] = mul0 + mul1;
            }
        }
    }
#endif

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_3d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
