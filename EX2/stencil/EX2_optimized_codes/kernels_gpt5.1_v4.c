#include "common.h"
#ifdef _OPENMP
#include <omp.h>
#endif

void cpu_stencil(float c0, float c1, float *A0, float *Anext, const int nx, const int ny, const int nz)
{
  int i, j, k;
  const int nxy = nx * ny;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) private(i, j, k) schedule(static)
#endif
  for (i = 1; i < nx - 1; i++) {
    for (j = 1; j < ny - 1; j++) {
      int base = i + nx * j;          /* (i, j, 0) plane index */
      int idx_center = base + nxy;    /* (i, j, 1) starting k=1 index */
      for (k = 1; k < nz - 1; k++) {
        Anext[idx_center] =
          ( A0[idx_center + 1]       /* (i, j, k+1)   */
          + A0[idx_center - 1]       /* (i, j, k-1)   */
          + A0[idx_center + nx]      /* (i, j+1, k)   */
          + A0[idx_center - nx]      /* (i, j-1, k)   */
          + A0[idx_center + nxy]     /* (i+1, j, k)   */
          + A0[idx_center - nxy] )   /* (i-1, j, k)   */
          * c1
          - A0[idx_center] * c0;     /* center (i, j, k) */

        idx_center += nxy;           /* advance to (i, j, k+1) */
      }
    }
  }
}


