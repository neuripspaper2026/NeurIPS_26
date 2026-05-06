#include "common.h"

void cpu_stencil(float c0, float c1, float *A0, float *Anext, const int nx, const int ny, const int nz)
{
  const int nxny = nx * ny;
  const int imax = nx - 1;
  const int jmax = ny - 1;
  const int kmax = nz - 1;

  for (int i = 1; i < imax; ++i) {
    const int iOff      = i * nxny;
    const int iOff_m1   = iOff - nxny;
    const int iOff_p1   = iOff + nxny;

    for (int j = 1; j < jmax; ++j) {
      const int base     = iOff + j * nx;
      const int base_mj1 = iOff + (j - 1) * nx;
      const int base_pj1 = iOff + (j + 1) * nx;

      /* k = 1: cannot reuse k-1 center value from previous iteration */
      {
        const int idx     = base + 1;
        const float center = A0[idx];
        const float xm     = A0[idx - 1];
        const float xp     = A0[idx + 1];
        const float ym     = A0[base_mj1 + 1];
        const float yp     = A0[base_pj1 + 1];
        const float zm     = A0[iOff_m1 + j * nx + 1];
        const float zp     = A0[iOff_p1 + j * nx + 1];

        Anext[idx] = (xp + xm + yp + ym + zp + zm) * c1 - center * c0;

        float prev_center = center;
        float prev_xp     = xp;

        /* k from 2 to kmax-1: reuse previous k as k-1 */
        for (int k = 2; k < kmax; ++k) {
          const int idx_k   = base + k;
          const float center_k = prev_xp;               /* A0(i,j,k)   */
          const float xp_k     = A0[idx_k + 1];         /* A0(i,j,k+1) */
          const float xm_k     = prev_center;           /* A0(i,j,k-1) */
          const float ym_k     = A0[base_mj1 + k];
          const float yp_k     = A0[base_pj1 + k];
          const float zm_k     = A0[iOff_m1 + j * nx + k];
          const float zp_k     = A0[iOff_p1 + j * nx + k];

          Anext[idx_k] = (xp_k + xm_k + yp_k + ym_k + zp_k + zm_k) * c1 - center_k * c0;

          prev_center = center_k;
          prev_xp     = xp_k;
        }
      }
    }
  }
}


