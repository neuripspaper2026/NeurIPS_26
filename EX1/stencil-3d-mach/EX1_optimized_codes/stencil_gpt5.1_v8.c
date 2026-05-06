#include <time.h>
#include "../stencil.h"

static double stencil_3d_kernel_time_acc = 0.0;

void reset_stencil_3d_kernel_time(void) { stencil_3d_kernel_time_acc = 0.0; }
double get_stencil_3d_kernel_time(void) { return stencil_3d_kernel_time_acc; }

void stencil3d(TYPE C[2], TYPE orig[SIZE], TYPE sol[SIZE]) {
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    // Cache local copies of constants for faster access
    const int rs = row_size;
    const int cs = col_size;
    const int hs = height_size;
    const int hs_1 = hs - 1;
    const int cs_1 = cs - 1;
    const int rs_1 = rs - 1;

    // Precompute plane and row strides for linear indexing
    const int plane_stride = rs * cs;
    const int row_stride   = rs;

    // Copy boundaries: height (k, j, 0 and k, j, hs-1)
    for (int j = 0; j < cs; ++j) {
        for (int k = 0; k < rs; ++k) {
            const int base_idx = k + row_stride * j;
            sol[base_idx] = orig[base_idx];
            const int top_idx = base_idx + plane_stride * hs_1;
            sol[top_idx] = orig[top_idx];
        }
    }

    // Copy boundaries: columns (k, 0, i and k, cs-1, i)
    for (int i = 1; i < hs_1; ++i) {
        const int plane_off = plane_stride * i;
        const int col0_off  = plane_off;             // j = 0
        const int colL_off  = plane_off + row_stride * cs_1; // j = cs-1
        for (int k = 0; k < rs; ++k) {
            const int idx0 = col0_off + k;
            const int idxL = colL_off + k;
            sol[idx0] = orig[idx0];
            sol[idxL] = orig[idxL];
        }
    }

    // Copy boundaries: rows (0, j, i and rs-1, j, i)
    for (int i = 1; i < hs_1; ++i) {
        const int plane_off = plane_stride * i;
        for (int j = 1; j < cs_1; ++j) {
            const int row_off = plane_off + row_stride * j;
            const int idx0 = row_off;        // k = 0
            const int idxR = row_off + rs_1; // k = rs-1
            sol[idx0] = orig[idx0];
            sol[idxR] = orig[idxR];
        }
    }

    // Stencil computation on interior points
    const TYPE c0 = C[0];
    const TYPE c1 = C[1];

    for (int i = 1; i < hs_1; ++i) {
        const int plane_off      = plane_stride * i;
        const int plane_off_above = plane_off + plane_stride;   // i + 1
        const int plane_off_below = plane_off - plane_stride;   // i - 1

        for (int j = 1; j < cs_1; ++j) {
            const int row_off      = plane_off + row_stride * j;
            const int row_off_north = row_off + row_stride;     // j + 1
            const int row_off_south = row_off - row_stride;     // j - 1

            // k loop over interior
            for (int k = 1; k < rs_1; ++k) {
                const int idx = row_off + k;

                const TYPE center = orig[idx];

                const TYPE sum_neighbors =
                    orig[idx + plane_stride] +   // (i + 1, j,     k)
                    orig[idx - plane_stride] +   // (i - 1, j,     k)
                    orig[row_off_north + k] +    // (i,     j + 1, k)
                    orig[row_off_south + k] +    // (i,     j - 1, k)
                    orig[idx + 1] +              // (i,     j,     k + 1)
                    orig[idx - 1];               // (i,     j,     k - 1)

                sol[idx] = center * c0 + sum_neighbors * c1;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_3d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
