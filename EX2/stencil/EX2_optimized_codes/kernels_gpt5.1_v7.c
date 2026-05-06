#include "common.h"
#ifdef _OPENMP
#include <omp.h>
#endif

void cpu_stencil(float c0,float c1, float *A0,float * Anext,const int nx, const int ny, const int nz)
{
  /* Precompute products used by Index3D to avoid repeated multiplies */
  const int nxy = nx * ny;

  int i, j, k;

#ifdef _OPENMP
  /* Parallelize outer loops; collapse to improve load balance and cache use */
  #pragma omp parallel for collapse(2) private(i,j,k) schedule(static)
#endif
  for(i = 1; i < nx - 1; ++i) {
    for(j = 1; j < ny - 1; ++j) {

      /* Base index for (i,j,0) plane; inner loop updates k dimension */
      int base_idx = i + nx * (j + ny * 0);

      /* Start at k=1, so initial center index is for k=1 */
      for(k = 1; k < nz - 1; ++k) {

        /* Current center index */
        int idx    = base_idx + nxy * k;
        int idx_pz = idx + nxy;   /* k + 1 */
        int idx_mz = idx - nxy;   /* k - 1 */
        int idx_py = idx + nx;    /* j + 1 */
        int idx_my = idx - nx;    /* j - 1 */
        int idx_px = idx + 1;     /* i + 1 */
        int idx_mx = idx - 1;     /* i - 1 */

        float center = A0[idx];

        float sum_neigh =
          A0[idx_pz] +
          A0[idx_mz] +
          A0[idx_py] +
          A0[idx_my] +
          A0[idx_px] +
          A0[idx_mx];

        Anext[idx] = sum_neigh * c1 - center * c0;
      }
    }
  }
}


