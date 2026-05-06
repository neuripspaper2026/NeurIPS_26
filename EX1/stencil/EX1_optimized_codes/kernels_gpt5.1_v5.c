#include "common.h"

void cpu_stencil(float c0,float c1, float *A0,float * Anext,const int nx, const int ny, const int nz)
{
  int i, j, k;

  const int nxny = nx * ny;

  for(i = 1; i < nx - 1; ++i)
  {
    const int ix_base = i * nxny;
    const int ixp_base = (i + 1) * nxny;
    const int ixm_base = (i - 1) * nxny;

    for(j = 1; j < ny - 1; ++j)
    {
      const int base = ix_base + j * nx;
      const int jp_base = ix_base + (j + 1) * nx;
      const int jm_base = ix_base + (j - 1) * nx;

      for(k = 1; k < nz - 1; ++k)
      {
        const int idx   = base + k;
        const int idx_pz = idx + 1;
        const int idx_mz = idx - 1;

        const float center = A0[idx];

        const float xp = A0[ixp_base + j * nx + k];
        const float xm = A0[ixm_base + j * nx + k];
        const float yp = A0[jp_base + k];
        const float ym = A0[jm_base + k];
        const float zp = A0[idx_pz];
        const float zm = A0[idx_mz];

        Anext[idx] = (xp + xm + yp + ym + zp + zm) * c1 - center * c0;
      }
    }
  }
}


