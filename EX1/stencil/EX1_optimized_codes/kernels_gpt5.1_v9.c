#include "common.h"

void cpu_stencil(float c0,float c1, float *A0,float * Anext,const int nx, const int ny, const int nz)
{
  const int nxny = nx * ny;
  const int i_start = 1;
  const int i_end = nx - 1;
  const int j_start = 1;
  const int j_end = ny - 1;
  const int k_start = 1;
  const int k_end = nz - 1;

  for (int i = i_start; i < i_end; ++i) {
    const int i_nxny = i * nxny;
    const int i_plus_nxny  = (i + 1) * nxny;
    const int i_minus_nxny = (i - 1) * nxny;

    for (int j = j_start; j < j_end; ++j) {
      const int j_nx = j * nx;
      const int j_plus_nx  = (j + 1) * nx;
      const int j_minus_nx = (j - 1) * nx;

      const int center_base  = i_nxny       + j_nx;
      const int north_base   = i_nxny       + j_plus_nx;
      const int south_base   = i_nxny       + j_minus_nx;
      const int up_base      = i_plus_nxny  + j_nx;
      const int down_base    = i_minus_nxny + j_nx;

      for (int k = k_start; k < k_end; ++k) {
        const int idx_center = center_base + k;

        const float center = A0[idx_center];

        const float xp = A0[up_base   + k];
        const float xm = A0[down_base + k];
        const float yp = A0[north_base + k];
        const float ym = A0[south_base + k];
        const float zp = A0[idx_center + 1];
        const float zm = A0[idx_center - 1];

        Anext[idx_center] = (xp + xm + yp + ym + zp + zm) * c1 - center * c0;
      }
    }
  }
}


