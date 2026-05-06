#include "common.h"

void cpu_stencil(float c0,float c1, float *A0,float * Anext,const int nx, const int ny, const int nz)
{
  const int nxny = nx * ny;

  for (int i = 1; i < nx - 1; ++i) {
    const int i_off = i * nxny;
    const int ip1_off = (i + 1) * nxny;
    const int im1_off = (i - 1) * nxny;

    for (int j = 1; j < ny - 1; ++j) {
      const int ij_off = i_off + j * nx;
      const int ijp1_off = i_off + (j + 1) * nx;
      const int ijm1_off = i_off + (j - 1) * nx;
      const int ip1j_off = ip1_off + j * nx;
      const int im1j_off = im1_off + j * nx;

      for (int k = 1; k < nz - 1; ++k) {
        const int idx    = ij_off  + k;
        const int idx_pz = idx + 1;
        const int idx_mz = idx - 1;

        const float center = A0[idx];
        const float xp = A0[ip1j_off + k];
        const float xm = A0[im1j_off + k];
        const float yp = A0[ijp1_off + k];
        const float ym = A0[ijm1_off + k];
        const float zp = A0[idx_pz];
        const float zm = A0[idx_mz];

        Anext[idx] = (xp + xm + yp + ym + zp + zm) * c1 - center * c0;
      }
    }
  }
}


