#include "common.h"

void cpu_stencil(float c0, float c1, float *A0, float *Anext, const int nx, const int ny, const int nz)
{
  int i, j, k;

  const int nxny = nx * ny;

  for (i = 1; i < nx - 1; i++) {
    for (j = 1; j < ny - 1; j++) {

      /* Precompute plane and row offsets once per (i, j) pair */
      const int base = Index3D(nx, ny, i, j, 0);
      const int idx_center = base + 1;                /* k = 1 */
      int idx = idx_center;                           /* running index for current (i,j,k) */

      const int stride_k  = 1;                        /* move along z */
      const int stride_j  = nx;                       /* move along y */
      const int stride_i  = nxny;                     /* move along x */

      /* Neighbors that change with k */
      float xm, xp, ym, yp, zm, zp, center;

      /* Initialize neighbors for k = 1 */
      xm = A0[idx - stride_i];
      xp = A0[idx + stride_i];
      ym = A0[idx - stride_j];
      yp = A0[idx + stride_j];
      zm = A0[idx - stride_k];
      zp = A0[idx + stride_k];
      center = A0[idx];

      for (k = 1; k < nz - 1; k++, idx += stride_k) {

        /* Write result for current center */
        Anext[idx] = (xm + xp + ym + yp + zm + zp) * c1 - center * c0;

        /* Prepare neighbors for next k (k+1), unless we're at the last iteration */
        if (k + 1 < nz - 1) {
          const int idx_next = idx + stride_k;

          /* shift zm to current center, get new zp */
          zm = center;
          center = A0[idx_next];
          zp = A0[idx_next + stride_k];

          /* xm, xp, ym, yp are invariant in k for fixed i,j so no need to recompute */
        }
      }
    }
  }
}


