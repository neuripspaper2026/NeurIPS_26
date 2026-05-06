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

    const int rs = row_size;
    const int cs = col_size;
    const int hs = height_size;

    const TYPE c0 = C[0];
    const TYPE c1 = C[1];

    // Precompute commonly used plane and row strides
    const int plane_stride = rs * cs;
    const int row_stride   = rs;

    // Handle boundary conditions by filling with original values
    height_bound_col:
    for (j = 0; j < cs; j++) {
        const int base0 = rs * (j + cs * 0);
        const int baseH = rs * (j + cs * (hs - 1));
        height_bound_row:
        for (k = 0; k < rs; k++) {
            sol[base0 + k] = orig[base0 + k];
            sol[baseH + k] = orig[baseH + k];
        }
    }

    col_bound_height:
    for (i = 1; i < hs - 1; i++) {
        const int base = plane_stride * i;
        const int base0 = base + rs * 0;
        const int baseC = base + rs * (cs - 1);
        col_bound_row:
        for (k = 0; k < rs; k++) {
            sol[base0 + k] = orig[base0 + k];
            sol[baseC + k] = orig[baseC + k];
        }
    }

    row_bound_height:
    for (i = 1; i < hs - 1; i++) {
        const int base = plane_stride * i;
        row_bound_col:
        for (j = 1; j < cs - 1; j++) {
            const int idx0 = base + rs * j + 0;
            const int idxR = base + rs * j + (rs - 1);
            sol[idx0] = orig[idx0];
            sol[idxR] = orig[idxR];
        }
    }

    // Stencil computation
    loop_height:
    for (i = 1; i < hs - 1; i++) {
        const int base_mid  = plane_stride * i;
        const int base_up   = plane_stride * (i + 1);
        const int base_down = plane_stride * (i - 1);

        loop_col:
        for (j = 1; j < cs - 1; j++) {
            const int row_off     = base_mid  + row_stride * j;
            const int row_off_up  = base_up   + row_stride * j;
            const int row_off_dn  = base_down + row_stride * j;
            const int row_off_pj  = base_mid  + row_stride * (j + 1);
            const int row_off_mj  = base_mid  + row_stride * (j - 1);

            loop_row:
            for (k = 1; k < rs - 1; k++) {
                const int idx      = row_off + k;
                const int idx_up   = row_off_up + k;      // (k, j, i+1)
                const int idx_dn   = row_off_dn + k;      // (k, j, i-1)
                const int idx_pj   = row_off_pj + k;      // (k, j+1, i)
                const int idx_mj   = row_off_mj + k;      // (k, j-1, i)
                const int idx_pk   = row_off + (k + 1);   // (k+1, j, i)
                const int idx_mk   = row_off + (k - 1);   // (k-1, j, i)

                sum0 = orig[idx];
                sum1 = orig[idx_up] +
                       orig[idx_dn] +
                       orig[idx_pj] +
                       orig[idx_mj] +
                       orig[idx_pk] +
                       orig[idx_mk];

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
