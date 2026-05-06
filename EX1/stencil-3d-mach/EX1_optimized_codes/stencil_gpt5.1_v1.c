#include <time.h>
#include "../stencil.h"

static double stencil_3d_kernel_time_acc = 0.0;

void reset_stencil_3d_kernel_time(void) { stencil_3d_kernel_time_acc = 0.0; }
double get_stencil_3d_kernel_time(void) { return stencil_3d_kernel_time_acc; }

void stencil3d(TYPE C[2], TYPE orig[SIZE], TYPE sol[SIZE]) {
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    // Handle boundary conditions by filling with original values
    {
        int j, k, i;

        for (j = 0; j < col_size; j++) {
            int base0 = row_size * (j + col_size * 0);
            int baseH = row_size * (j + col_size * (height_size - 1));
            for (k = 0; k < row_size; k++) {
                int idx0 = base0 + k;
                int idxH = baseH + k;
                sol[idx0] = orig[idx0];
                sol[idxH] = orig[idxH];
            }
        }

        for (i = 1; i < height_size - 1; i++) {
            int baseL = row_size * (0 + col_size * i);
            int baseR = row_size * ((col_size - 1) + col_size * i);
            for (k = 0; k < row_size; k++) {
                int idxL = baseL + k;
                int idxR = baseR + k;
                sol[idxL] = orig[idxL];
                sol[idxR] = orig[idxR];
            }
        }

        for (i = 1; i < height_size - 1; i++) {
            for (j = 1; j < col_size - 1; j++) {
                int base = row_size * (j + col_size * i);
                int idxF = base + 0;
                int idxB = base + (row_size - 1);
                sol[idxF] = orig[idxF];
                sol[idxB] = orig[idxB];
            }
        }
    }

    // Stencil computation
    {
        const TYPE c0 = C[0];
        const TYPE c1 = C[1];
        int i, j, k;

        for (i = 1; i < height_size - 1; i++) {
            int plane_offset = row_size * col_size * i;
            int plane_offset_p = plane_offset + row_size * col_size;
            int plane_offset_m = plane_offset - row_size * col_size;

            for (j = 1; j < col_size - 1; j++) {
                int row_offset = plane_offset + row_size * j;
                int row_offset_p = row_offset + row_size;
                int row_offset_m = row_offset - row_size;

                for (k = 1; k < row_size - 1; k++) {
                    int center = row_offset + k;

                    TYPE sum0 = orig[center];
                    TYPE sum1 =
                        orig[plane_offset_p + k] +
                        orig[plane_offset_m + k] +
                        orig[row_offset_p + k] +
                        orig[row_offset_m + k] +
                        orig[row_offset + (k + 1)] +
                        orig[row_offset + (k - 1)];

                    sol[center] = sum0 * c0 + sum1 * c1;
                }
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_3d_kernel_time_acc +=
        (double)(kernel_end.tv_sec - kernel_start.tv_sec) +
        (double)(kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
