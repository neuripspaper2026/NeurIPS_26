#include "common.h"

void cpu_stencil(float c0,float c1, float *A0,float * Anext,const int nx, const int ny, const int nz)
{
  int i, j, k;

  const int stride_x = 1;
  const int stride_y = nx;
  const int stride_z = nx * ny;

  for (i = 1; i < nx - 1; ++i) {
    const int base_i = i * stride_x;
    const int base_ip = (i + 1) * stride_x;
    const int base_im = (i - 1) * stride_x;

    for (j = 1; j < ny - 1; ++j) {
      const int base_ij   = base_i  + j * stride_y;
      const int base_ijp  = base_i  + (j + 1) * stride_y;
      const int base_ijm  = base_i  + (j - 1) * stride_y;
      const int base_ipj  = base_ip + j * stride_y;
      const int base_imj  = base_im + j * stride_y;

      for (k = 1; k < nz - 1; ++k) {
        const int idx   = base_ij  + k * stride_z;
        const int idx_pz = idx + stride_z;
        const int idx_mz = idx - stride_z;

        const float center = A0[idx];

        const float neighbor_sum =
          A0[idx_pz] +
          A0[idx_mz] +
          A0[base_ijp + k * stride_z] +
          A0[base_ijm + k * stride_z] +
          A0[base_ipj + k * stride_z] +
          A0[base_imj + k * stride_z];

        Anext[idx] = neighbor_sum * c1 - center * c0;
      }
    }
  }
}


