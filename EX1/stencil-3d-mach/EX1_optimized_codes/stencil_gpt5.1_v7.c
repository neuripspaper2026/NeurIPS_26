#include <time.h>
#include "../stencil.h"

static double stencil_3d_kernel_time_acc = 0.0;

void reset_stencil_3d_kernel_time(void) { stencil_3d_kernel_time_acc = 0.0; }
double get_stencil_3d_kernel_time(void) { return stencil_3d_kernel_time_acc; }

void stencil3d(TYPE C[2], TYPE orig[SIZE], TYPE sol[SIZE]) {
    int i, j, k;
    TYPE c0 = C[0];
    TYPE c1 = C[1];
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    // Handle boundary conditions by filling with original values
    for (j = 0; j < col_size; j++) {
        int base0 = j * row_size;
        int baseH = base0 + col_size * row_size * (height_size - 1);
        for (k = 0; k < row_size; k++) {
            int idx0 = base0 + k;
            int idxH = baseH + k;
            sol[idx0] = orig[idx0];
            sol[idxH] = orig[idxH];
        }
    }

    for (i = 1; i < height_size - 1; i++) {
        int plane_offset = i * col_size * row_size;
        for (k = 0; k < row_size; k++) {
            int idx_left  = plane_offset + k;
            int idx_right = plane_offset + k + row_size * (col_size - 1);
            sol[idx_left]  = orig[idx_left];
            sol[idx_right] = orig[idx_right];
        }
    }

    for (i = 1; i < height_size - 1; i++) {
        int plane_offset = i * col_size * row_size;
        for (j = 1; j < col_size - 1; j++) {
            int idx_front = plane_offset + j * row_size;
            int idx_back  = plane_offset + j * row_size + (row_size - 1);
            sol[idx_front] = orig[idx_front];
            sol[idx_back]  = orig[idx_back];
        }
    }

    // Stencil computation
    for (i = 1; i < height_size - 1; i++) {
        int plane_offset      = i * col_size * row_size;
        int plane_offset_next = (i + 1) * col_size * row_size;
        int plane_offset_prev = (i - 1) * col_size * row_size;

        for (j = 1; j < col_size - 1; j++) {
            int row_offset      = plane_offset      + j * row_size;
            int row_offset_next = plane_offset      + (j + 1) * row_size;
            int row_offset_prev = plane_offset      + (j - 1) * row_size;

            for (k = 1; k < row_size - 1; k++) {
                int center = row_offset + k;

                TYPE sum0 = orig[center];
                TYPE sum1 =
                    orig[plane_offset_next + j * row_size + k] +  // i+1
                    orig[plane_offset_prev + j * row_size + k] +  // i-1
                    orig[row_offset_next + k] +                   // j+1
                    orig[row_offset_prev + k] +                   // j-1
                    orig[row_offset + (k + 1)] +                  // k+1
                    orig[row_offset + (k - 1)];                   // k-1

                sol[center] = sum0 * c0 + sum1 * c1;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    stencil_3d_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
