#include <time.h>
#include "../stencil.h"

static double stencil_3d_kernel_time_acc = 0.0;

void reset_stencil_3d_kernel_time(void) { stencil_3d_kernel_time_acc = 0.0; }
double get_stencil_3d_kernel_time(void) { return stencil_3d_kernel_time_acc; }

void stencil3d(TYPE C[2], TYPE orig[SIZE], TYPE sol[SIZE]) {
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    // Precompute constants and strides
    const int rs = row_size;
    const int cs = col_size;
    const int hs = height_size;
    const int cs_mul = rs * cs;
    const int cs_off = rs * cs;
    const int row_last = rs - 1;
    const int col_last = cs - 1;
    const int h_last  = hs - 1;

    const TYPE c0 = C[0];
    const TYPE c1 = C[1];

    // Handle boundary conditions by copying from original
    for (int j = 0; j < cs; ++j) {
        int base0 = rs * (j + cs * 0);
        int baseH = rs * (j + cs * h_last);
        for (int k = 0; k < rs; ++k) {
            sol[base0 + k] = orig[base0 + k];
            sol[baseH + k] = orig[baseH + k];
        }
    }

    for (int i = 1; i < h_last; ++i) {
        int baseI  = rs * cs * i;
        int baseI0 = rs * (0 + cs * i);
        int baseIC = rs * (col_last + cs * i);
        for (int k = 0; k < rs; ++k) {
            sol[baseI0 + k] = orig[baseI0 + k];
            sol[baseIC + k] = orig[baseIC + k];
        }
    }

    for (int i = 1; i < h_last; ++i) {
        int baseI = rs * cs * i;
        for (int j = 1; j < col_last; ++j) {
            int baseIJ = baseI + rs * j;
            sol[baseIJ + 0]        = orig[baseIJ + 0];
            sol[baseIJ + row_last] = orig[baseIJ + row_last];
        }
    }

    // Stencil computation for interior points
    for (int i = 1; i < h_last; ++i) {
        int base_mid   = cs_mul * i;
        int base_above = base_mid + cs_off;
        int base_below = base_mid - cs_off;

        for (int j = 1; j < col_last; ++j) {
            int row_base_mid   = base_mid   + rs * j;
            int row_base_above = base_above + rs * j;
            int row_base_below = base_below + rs * j;
            int row_base_north = row_base_mid + rs;
            int row_base_south = row_base_mid - rs;

            // unroll inner loop by 2 for better ILP
            int k = 1;
            int k_end = row_last - 1;
            for (; k + 1 <= k_end; k += 2) {
                int idx0 = row_base_mid + k;
                int idx1 = idx0 + 1;

                TYPE center0 = orig[idx0];
                TYPE center1 = orig[idx1];

                TYPE sum1_0 =
                    orig[row_base_above + k] +
                    orig[row_base_below + k] +
                    orig[row_base_north + k] +
                    orig[row_base_south + k] +
                    orig[idx0 + 1] +
                    orig[idx0 - 1];

                TYPE sum1_1 =
                    orig[row_base_above + k + 1] +
                    orig[row_base_below + k + 1] +
                    orig[row_base_north + k + 1] +
                    orig[row_base_south + k + 1] +
                    orig[idx1 + 1] +
                    orig[idx1 - 1];

                sol[idx0] = center0 * c0 + sum1_0 * c1;
                sol[idx1] = center1 * c0 + sum1_1 * c1;
            }

            // handle remaining element if row_size-2 is odd
            for (; k <= k_end; ++k) {
                int idx = row_base_mid + k;
                TYPE center = orig[idx];
                TYPE sum1 =
                    orig[row_base_above + k] +
                    orig[row_base_below + k] +
                    orig[row_base_mid + rs + k] +
                    orig[row_base_mid - rs + k] +
                    orig[idx + 1] +
                    orig[idx - 1];

                sol[idx] = center * c0 + sum1 * c1;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_3d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
