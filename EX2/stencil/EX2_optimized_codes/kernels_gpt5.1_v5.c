#include "common.h"
#ifdef _OPENMP
#include <omp.h>
#endif

void cpu_stencil(float c0, float c1, float *A0, float *Anext,
                 const int nx, const int ny, const int nz)
{
  const int nxny = nx * ny;

  /* Parallelize outer loops if OpenMP is available.
     Use collapse(2) to distribute i,j iteration space among threads. */
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) default(none) \
    shared(A0, Anext, c0, c1, nx, ny, nz, nxny)
#endif
  for (int i = 1; i < nx - 1; i++) {
    for (int j = 1; j < ny - 1; j++) {

      /* Precompute plane offset for this (i,j) */
      const int base = i + nx * (j + ny * 0); /* k will be added below */

      /* Compute neighbor plane base indices once per (i,j) */
      const int base_im1 = base - 1;
      const int base_ip1 = base + 1;
      const int base_jm1 = base - nx;
      const int base_jp1 = base + nx;

      /* Start at k=1; we will use direct linear indexing for speed */
      for (int k = 1; k < nz - 1; k++) {
        const int idx    = base + nxny * k;
        const int idx_kp = idx + nxny;
        const int idx_km = idx - nxny;

        const float center = A0[idx];

        const float sum_nb =
          A0[idx_kp] + A0[idx_km] +
          A0[base_jp1 + nxny * k] +
          A0[base_jm1 + nxny * k] +
          A0[base_ip1 + nxny * k] +
          A0[base_im1 + nxny * k];

        Anext[idx] = sum_nb * c1 - center * c0;
      }
    }
  }
}


