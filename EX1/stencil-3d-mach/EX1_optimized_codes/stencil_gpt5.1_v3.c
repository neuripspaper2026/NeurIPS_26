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

    // Handle boundary conditions by filling with original values
    {
        int const rs = row_size;
        int const cs = col_size;
        int const hs = height_size;
        int const plane = rs * cs;

        // z (height) boundaries
        for (j = 0; j < cs; j++) {
            int base0 = rs * j;              // offset for k, z = 0
            int baseN = base0 + plane * (hs - 1); // offset for k, z = hs-1
            for (k = 0; k < rs; k++) {
                int idx0 = base0 + k;
                int idxN = baseN + k;
                sol[idx0] = orig[idx0];
                sol[idxN] = orig[idxN];
            }
        }

        // y (col) boundaries for inner z
        for (i = 1; i < hs - 1; i++) {
            int zoff = plane * i;
            for (k = 0; k < rs; k++) {
                int idx0 = zoff + k;               // j = 0
                int idxN = zoff + k + rs * (cs - 1); // j = cs-1
                sol[idx0] = orig[idx0];
                sol[idxN] = orig[idxN];
            }
        }

        // x (row) boundaries for inner z and y
        for (i = 1; i < hs - 1; i++) {
            int zoff = plane * i;
            for (j = 1; j < cs - 1; j++) {
                int base = zoff + rs * j;
                int idx0 = base;          // k = 0
                int idxN = base + (rs - 1); // k = rs-1
                sol[idx0] = orig[idx0];
                sol[idxN] = orig[idxN];
            }
        }
    }

    // Stencil computation
    {
        int const rs = row_size;
        int const cs = col_size;
        int const hs = height_size;
        int const plane = rs * cs;

        for (i = 1; i < hs - 1; i++) {
            int zoff    = plane * i;
            int zoff_p1 = zoff + plane;
            int zoff_m1 = zoff - plane;

            for (j = 1; j < cs - 1; j++) {
                int rowoff    = zoff + rs * j;
                int rowoff_p1 = rowoff + rs;
                int rowoff_m1 = rowoff - rs;

                // Precompute neighbor row offsets for +/- y
                for (k = 1; k < rs - 1; k++) {
                    int idx = rowoff + k;

                    sum0 = orig[idx];

                    sum1 = orig[rowoff_p1 + k] +  // j+1
                           orig[rowoff_m1 + k] +  // j-1
                           orig[zoff_p1 + rs * j + k] +  // i+1
                           orig[zoff_m1 + rs * j + k] +  // i-1
                           orig[rowoff + (k + 1)] +      // k+1
                           orig[rowoff + (k - 1)];       // k-1

                    mul0 = sum0 * C[0];
                    mul1 = sum1 * C[1];
                    sol[idx] = mul0 + mul1;
                }
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_3d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
