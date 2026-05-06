#include "common.h"

void cpu_stencil(float c0, float c1, float *A0, float *Anext, const int nx, const int ny, const int nz)
{
  int i, j, k;

  const int ny_nz = ny * nz;

  for (i = 1; i < nx - 1; i++) {
    const int base_i      = i * ny_nz;
    const int base_ip1    = (i + 1) * ny_nz;
    const int base_im1    = (i - 1) * ny_nz;

    for (j = 1; j < ny - 1; j++) {
      const int base_ij   = base_i   + j * nz;
      const int base_ijp1 = base_i   + (j + 1) * nz;
      const int base_ijm1 = base_i   + (j - 1) * nz;

      for (k = 1; k < nz - 1; k++) {
        const int idx    = base_ij   + k;
        const int idx_kp = idx + 1;
        const int idx_km = idx - 1;

        const float center = A0[idx];

        const float sum_nb =
            A0[idx_kp] +
            A0[idx_km] +
            A0[base_ijp1 + k] +
            A0[base_ijm1 + k] +
            A0[base_ip1 + j * nz + k] +
            A0[base_im1 + j * nz + k];

        Anext[idx] = sum_nb * c1 - center * c0;
      }
    }
  }
}


