#include "common.h"
#ifdef _OPENMP
#include <omp.h>
#endif

void cpu_stencil(float c0, float c1, float * __restrict__ A0, float * __restrict__ Anext,
                 const int nx, const int ny, const int nz)
{
  const int nxny = nx * ny;

#if defined(_OPENMP)
#pragma omp parallel
  {
#endif
    int i, j, k;
#if defined(_OPENMP)
    /* Collapse the outer two loops to improve load balance and cache locality */
#pragma omp for collapse(2) schedule(static)
#endif
    for (i = 1; i < nx - 1; i++) {
      for (j = 1; j < ny - 1; j++) {

        /* Precompute the base index for this (i, j) line */
        int base = i + nx * (j + ny * 1); /* k will start from 1 */

        /* Neighbor offsets in linear memory (k±1, j±1, i±1) */
        const int off_kp = nxny;   /* +1 in k   */
        const int off_km = -nxny;  /* -1 in k   */
        const int off_jp = nx;     /* +1 in j   */
        const int off_jm = -nx;    /* -1 in j   */
        const int off_ip = 1;      /* +1 in i   */
        const int off_im = -1;     /* -1 in i   */

        for (k = 1; k < nz - 1; k++) {
          const int idx = base + (k * nxny - nxny); /* equivalent to Index3D(nx,ny,i,j,k) */

          const float center = A0[idx];
          const float sum_nb =
              A0[idx + off_kp] +
              A0[idx + off_km] +
              A0[idx + off_jp] +
              A0[idx + off_jm] +
              A0[idx + off_ip] +
              A0[idx + off_im];

          Anext[idx] = sum_nb * c1 - center * c0;
        }
      }
    }
#if defined(_OPENMP)
  }
#endif
}


