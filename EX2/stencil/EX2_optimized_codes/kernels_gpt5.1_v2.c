#include "common.h"
#ifdef _OPENMP
#include <omp.h>
#endif

void cpu_stencil(float c0,float c1, float *A0,float * Anext,const int nx, const int ny, const int nz)
{
  /* Hoist common products to avoid repeated integer multiplications */
  const int nxy = nx * ny;

  /* Simple bounds check: if any dimension is too small, there is no interior */
  if (nx <= 2 || ny <= 2 || nz <= 2) {
    return;
  }

  int i, j, k;

  /* Parallelize outer loops with OpenMP and improve locality by computing
   * per-(i,j) base indices. */
#ifdef _OPENMP
#pragma omp parallel for collapse(2) private(i,j,k)
#endif
  for (i = 1; i < nx - 1; i++) {
    for (j = 1; j < ny - 1; j++) {

      /* Precompute base index for (i,j,0) to simplify indexing */
      const int base_ij = i + nx * j;

      for (k = 1; k < nz - 1; k++) {
        const int idx    = base_ij + nxy * k;
        const int idx_pz = idx + nxy;
        const int idx_mz = idx - nxy;
        const int idx_py = idx + nx;
        const int idx_my = idx - nx;
        const int idx_px = idx + 1;
        const int idx_mx = idx - 1;

        const float center = A0[idx];

        Anext[idx] =
          ( A0[idx_pz] +
            A0[idx_mz] +
            A0[idx_py] +
            A0[idx_my] +
            A0[idx_px] +
            A0[idx_mx]) * c1
          - center * c0;
      }
    }
  }
}


