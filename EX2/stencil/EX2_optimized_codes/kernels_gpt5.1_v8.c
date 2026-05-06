#include "common.h"
#ifdef _OPENMP
#include <omp.h>
#endif

void cpu_stencil(float c0,float c1, float *A0,float * Anext,const int nx, const int ny, const int nz)
{
  const int nxny = nx * ny;
  int i, j, k;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) private(i,j,k) schedule(static)
#endif
  for(i = 1; i < nx - 1; i++) {
    for(j = 1; j < ny - 1; j++) {

      const int base = i + nx * j;          /* (i,j,0) linear base */
      const int idx_center = base + nxny;   /* (i,j,1) */
      const int stride_k = nxny;            /* step in z */
      const int stride_i = 1;               /* step in x */
      const int stride_j = nx;              /* step in y */

      /* k = 1 separately to avoid extra loads of neighbors reused later (optional micro-opt) */
      int idx = idx_center;
      for(k = 1; k < nz - 1; k++, idx += stride_k) {

        const int idx_pz = idx + stride_k;
        const int idx_mz = idx - stride_k;
        const int idx_py = idx + stride_j;
        const int idx_my = idx - stride_j;
        const int idx_px = idx + stride_i;
        const int idx_mx = idx - stride_i;

        const float center = A0[idx];
        const float sum_neighbors =
            A0[idx_pz] +
            A0[idx_mz] +
            A0[idx_py] +
            A0[idx_my] +
            A0[idx_px] +
            A0[idx_mx];

        Anext[idx] = sum_neighbors * c1 - center * c0;
      }
    }
  }
}


