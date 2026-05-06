#include "common.h"

void cpu_stencil(float c0,float c1, float *A0,float * Anext,const int nx, const int ny, const int nz)
{
  const int nxny = nx * ny;

  for (int k = 1; k < nz - 1; ++k) {
    const int kOffset      = k * nxny;
    const int kOffsetPlus  = (k + 1) * nxny;
    const int kOffsetMinus = (k - 1) * nxny;

    for (int j = 1; j < ny - 1; ++j) {
      const int jOffset      = j * nx;
      const int jOffsetPlus  = (j + 1) * nx;
      const int jOffsetMinus = (j - 1) * nx;

      const int baseCenter   = kOffset + jOffset;
      const int baseKPlus    = kOffsetPlus + jOffset;
      const int baseKMinus   = kOffsetMinus + jOffset;
      const int baseJPlus    = kOffset + jOffsetPlus;
      const int baseJMinus   = kOffset + jOffsetMinus;

      for (int i = 1; i < nx - 1; ++i) {
        const int idxCenter = baseCenter + i;

        const float center = A0[idxCenter];

        const float xp = A0[idxCenter + 1];
        const float xm = A0[idxCenter - 1];

        const float yp = A0[baseJPlus  + i];
        const float ym = A0[baseJMinus + i];

        const float zp = A0[baseKPlus  + i];
        const float zm = A0[baseKMinus + i];

        Anext[idxCenter] = (xp + xm + yp + ym + zp + zm) * c1 - center * c0;
      }
    }
  }
}


