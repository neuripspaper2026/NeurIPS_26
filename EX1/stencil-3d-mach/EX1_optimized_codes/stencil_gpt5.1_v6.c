#include <time.h>
#include "../stencil.h"

static double stencil_3d_kernel_time_acc = 0.0;

void reset_stencil_3d_kernel_time(void) { stencil_3d_kernel_time_acc = 0.0; }
double get_stencil_3d_kernel_time(void) { return stencil_3d_kernel_time_acc; }

void stencil3d(TYPE C[2], TYPE orig[SIZE], TYPE sol[SIZE]) {
    int i, j, k;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    const int rs = row_size;
    const int cs = col_size;
    const int hs = height_size;
    const int rs_cs = rs * cs;
    const TYPE c0 = C[0];
    const TYPE c1 = C[1];

    // Handle boundary conditions by filling with original values
    for (j = 0; j < cs; j++) {
        for (k = 0; k < rs; k++) {
            const int base0 = k + rs * (j + cs * 0);
            const int base1 = k + rs * (j + cs * (hs - 1));
            sol[base0] = orig[base0];
            sol[base1] = orig[base1];
        }
    }

    for (i = 1; i < hs - 1; i++) {
        const int z_off = i * rs_cs;
        for (k = 0; k < rs; k++) {
            int base = z_off + k;              // (i, j=0, k)
            int base_last = z_off + k + rs * (cs - 1); // (i, j=cs-1, k)
            sol[base] = orig[base];
            sol[base_last] = orig[base_last];
        }
    }

    for (i = 1; i < hs - 1; i++) {
        const int z_off = i * rs_cs;
        for (j = 1; j < cs - 1; j++) {
            const int y_off = j * rs;
            const int base0 = z_off + y_off;                // (i, j, k=0)
            const int base1 = z_off + y_off + (rs - 1);     // (i, j, k=rs-1)
            sol[base0] = orig[base0];
            sol[base1] = orig[base1];
        }
    }

    // Stencil computation
    for (i = 1; i < hs - 1; i++) {
        const int z_off     = i * rs_cs;
        const int z_off_p1  = (i + 1) * rs_cs;
        const int z_off_m1  = (i - 1) * rs_cs;
        for (j = 1; j < cs - 1; j++) {
            const int y_off     = j * rs;
            const int y_off_p1  = (j + 1) * rs;
            const int y_off_m1  = (j - 1) * rs;

            const int base      = z_off    + y_off;
            const int base_zp1  = z_off_p1 + y_off;
            const int base_zm1  = z_off_m1 + y_off;
            const int base_yp1  = z_off    + y_off_p1;
            const int base_ym1  = z_off    + y_off_m1;

            // pointer aliases for this (i,j) line in k
            TYPE *orig_c   = orig + base;
            TYPE *orig_zp1 = orig + base_zp1;
            TYPE *orig_zm1 = orig + base_zm1;
            TYPE *orig_yp1 = orig + base_yp1;
            TYPE *orig_ym1 = orig + base_ym1;
            TYPE *sol_c    = sol  + base;

            for (k = 1; k < rs - 1; k++) {
                const TYPE center = orig_c[k];
                const TYPE sum1 =
                    orig_zp1[k] + orig_zm1[k] +
                    orig_yp1[k] + orig_ym1[k] +
                    orig_c[k + 1] + orig_c[k - 1];
                sol_c[k] = center * c0 + sum1 * c1;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_3d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
