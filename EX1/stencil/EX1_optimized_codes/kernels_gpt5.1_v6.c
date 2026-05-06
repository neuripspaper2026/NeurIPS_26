#include "common.h"

void cpu_stencil(float c0, float c1, float *A0, float *Anext, const int nx, const int ny, const int nz)
{
  const int nxny = nx * ny;
  const int nx_minus_1 = nx - 1;
  const int ny_minus_1 = ny - 1;
  const int nz_minus_1 = nz - 1;

  for (int k = 1; k < nz_minus_1; ++k) {
    const int base_k    = k * nxny;
    const int base_kp1  = base_k + nxny;
    const int base_km1  = base_k - nxny;

    for (int j = 1; j < ny_minus_1; ++j) {
      const int base_j   = base_k   + j * nx;
      const int base_jp1 = base_j   + nx;
      const int base_jm1 = base_j   - nx;
      const int base_kp1j = base_kp1 + j * nx;
      const int base_kmj  = base_km1 + j * nx;

      int idx = base_j + 1;
      const int idx_end = base_j + nx_minus_1;

      float xm1 = A0[idx - 1];
      float xc  = A0[idx];
      float xp1 = A0[idx + 1];

      for (; idx < idx_end; ++idx) {
        const int off = idx - base_j;

        const float yp1 = A0[base_jp1 + off];
        const float ym1 = A0[base_jm1 + off];
        const float zp1 = A0[base_kp1j + off];
        const float zm1 = A0[base_kmj  + off];

        const float center = xc;

        const float val = (xp1 + xm1 + yp1 + ym1 + zp1 + zm1) * c1 - center * c0;
        Anext[idx] = val;

        xm1 = xc;
        xc  = xp1;
        xp1 = A0[idx + 1];
      }
    }
  }
}


