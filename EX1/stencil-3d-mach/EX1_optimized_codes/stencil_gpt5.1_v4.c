#include <time.h>
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
    const int rs_cs = rs * cs;

    // Precompute coefficient values
    const TYPE c0 = C[0];
    const TYPE c1 = C[1];

    // Handle boundary conditions by filling with original values
    // k is fastest varying index; compute linear index directly
    for (int j = 0; j < cs; j++) {
        int base0 = rs * (j + cs * 0);
        int baseH = rs * (j + cs * (hs - 1));
        for (int k = 0; k < rs; k++) {
            sol[base0 + k] = orig[base0 + k];
            sol[baseH + k] = orig[baseH + k];
        }
    }

    for (int i = 1; i < hs - 1; i++) {
        int base = rs_cs * i;
        for (int k = 0; k < rs; k++) {
            int idxL = base + rs * 0 + k;
            int idxR = base + rs * (cs - 1) + k;
            sol[idxL] = orig[idxL];
            sol[idxR] = orig[idxR];
        }
    }

    for (int i = 1; i < hs - 1; i++) {
        int base = rs_cs * i;
        for (int j = 1; j < cs - 1; j++) {
            int idxF = base + rs * j + 0;
            int idxB = base + rs * j + (rs - 1);
            sol[idxF] = orig[idxF];
            sol[idxB] = orig[idxB];
        }
    }

    // Stencil computation
    for (int i = 1; i < hs - 1; i++) {
        int base   = rs_cs * i;
        int base_p = base + rs_cs;   // i+1
        int base_m = base - rs_cs;   // i-1
        for (int j = 1; j < cs - 1; j++) {
            int row   = base   + rs * j;
            int row_p = base   + rs * (j + 1);
            int row_m = base   + rs * (j - 1);
            int row_ip = base_p + rs * j;
            int row_im = base_m + rs * j;

            // k loop is contiguous; use direct indexing
            for (int k = 1; k < rs - 1; k++) {
                int idx = row + k;

                TYPE center = orig[idx];
                TYPE sum1 =
                    orig[row_ip + k] +      // i+1
                    orig[row_im + k] +      // i-1
                    orig[row_p  + k] +      // j+1
                    orig[row_m  + k] +      // j-1
                    orig[idx + 1] +         // k+1
                    orig[idx - 1];          // k-1

                sol[idx] = center * c0 + sum1 * c1;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_3d_kernel_time_acc +=
        (kernel_end.tv_sec - kernel_start.tv_sec) +
        (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
