#include "common.h"
#ifdef _OPENMP
#include <omp.h>
#endif

void cpu_stencil(float c0,float c1, float *A0,float * Anext,const int nx, const int ny, const int nz)
{
  /* Precompute constants to reduce repeated integer multiplications */
  const int ny_nz = ny * nz;
  const int nx_ny = nx * ny;

  int i, j, k;

#ifdef _OPENMP
  /* Parallelize over the outermost loop and collapse j, k for better load balance.
     Use static scheduling for regular iteration space. */
  #pragma omp parallel for collapse(2) private(i,j,k) schedule(static)
#endif
  for(i = 1; i < nx - 1; i++) {
    for(j = 1; j < ny - 1; j++) {

      /* Compute base index for A0 and Anext at k = 1 for this (i,j).
         We will update it incrementally inside the k loop. */
      int base = i * ny_nz + j * nz;

      for(k = 1; k < nz - 1; k++) {

        const int center = base + k;

        const float center_val = A0[center];
        const float xp = A0[center + ny_nz]; /* i + 1 */
        const float xm = A0[center - ny_nz]; /* i - 1 */
        const float yp = A0[center + nz];    /* j + 1 */
        const float ym = A0[center - nz];    /* j - 1 */
        const float zp = A0[center + 1];     /* k + 1 */
        const float zm = A0[center - 1];     /* k - 1 */

        Anext[center] = (xp + xm + yp + ym + zp + zm) * c1 - center_val * c0;
      }
    }
  }
}


