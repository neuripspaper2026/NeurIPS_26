#include <time.h>
#include "../stencil.h"

static double stencil_3d_kernel_time_acc = 0.0;

void reset_stencil_3d_kernel_time(void) { stencil_3d_kernel_time_acc = 0.0; }
double get_stencil_3d_kernel_time(void) { return stencil_3d_kernel_time_acc; }

void stencil3d(TYPE C[2], TYPE orig[SIZE], TYPE sol[SIZE]) {
  struct timespec kernel_start, kernel_end;
  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  const int rs  = row_size;
  const int cs  = col_size;
  const int hs  = height_size;
  const int cs_hs = cs * hs;
  const int plane = cs * rs;

  TYPE c0 = C[0];
  TYPE c1 = C[1];

  // Handle boundary conditions by copying original values
  for (int j = 0; j < cs; ++j) {
    int base0 = j * rs;
    int baseH = base0 + (hs - 1) * plane;
    for (int k = 0; k < rs; ++k) {
      sol[base0 + k] = orig[base0 + k];
      sol[baseH + k] = orig[baseH + k];
    }
  }

  for (int i = 1; i < hs - 1; ++i) {
    int plane_i = i * plane;
    for (int k = 0; k < rs; ++k) {
      int idx0 = plane_i + k;
      int idxC = idx0 + (0 * rs);
      int idxL = idx0 + ((cs - 1) * rs);
      sol[idxC] = orig[idxC];
      sol[idxL] = orig[idxL];
    }
  }

  for (int i = 1; i < hs - 1; ++i) {
    int plane_i = i * plane;
    for (int j = 1; j < cs - 1; ++j) {
      int base = plane_i + j * rs;
      sol[base + 0]      = orig[base + 0];
      sol[base + rs - 1] = orig[base + rs - 1];
    }
  }

  // Stencil computation
  for (int i = 1; i < hs - 1; ++i) {
    int plane_im1 = (i - 1) * plane;
    int plane_i   = i * plane;
    int plane_ip1 = (i + 1) * plane;

    for (int j = 1; j < cs - 1; ++j) {
      int row_base_im1 = plane_im1 + j * rs;
      int row_base_i   = plane_i   + j * rs;
      int row_base_ip1 = plane_ip1 + j * rs;

      int row_base_i_jm1 = row_base_i - rs;
      int row_base_i_jp1 = row_base_i + rs;

      for (int k = 1; k < rs - 1; ++k) {
        int center_idx = row_base_i + k;

        TYPE center = orig[center_idx];

        TYPE sum1 =
          orig[row_base_ip1 + k] +  // (i+1,j,k)
          orig[row_base_im1 + k] +  // (i-1,j,k)
          orig[row_base_i_jp1 + k] +// (i,j+1,k)
          orig[row_base_i_jm1 + k] +// (i,j-1,k)
          orig[center_idx + 1] +    // (i,j,k+1)
          orig[center_idx - 1];     // (i,j,k-1)

        sol[center_idx] = center * c0 + sum1 * c1;
      }
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  stencil_3d_kernel_time_acc +=
      (kernel_end.tv_sec - kernel_start.tv_sec) +
      (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
