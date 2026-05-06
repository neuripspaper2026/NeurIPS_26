#include "common.h"
#ifdef _OPENMP
#include <omp.h>
#endif

void cpu_stencil(float c0,float c1, float *A0,float * Anext,const int nx, const int ny, const int nz)
{
  int i, j, k;

  /* Precompute plane and row strides for better cache and index reuse */
  const int stride_y = nx;
  const int stride_z = nx * ny;

  /* Parallelize the outer loops with OpenMP, and collapse j-k to increase work per thread.
     Keep default scheduling (implementation may tune it) and avoid false sharing by using
     separate indices per thread. */
#ifdef _OPENMP
#pragma omp parallel for private(j,k) schedule(static) collapse(2)
#endif
  for (i = 1; i < nx - 1; i++) {
    for (j = 1; j < ny - 1; j++) {
      /* Compute base index for this (i,j) row once */
      int base = i + stride_y * j;
      for (k = 1; k < nz - 1; k++) {
        int idx   = base + stride_z * k;
        int idx_pz = idx + stride_z;
        int idx_mz = idx - stride_z;
        int idx_py = idx + stride_y;
        int idx_my = idx - stride_y;
        int idx_px = idx + 1;
        int idx_mx = idx - 1;

        Anext[idx] = (A0[idx_pz] +
                      A0[idx_mz] +
                      A0[idx_py] +
                      A0[idx_my] +
                      A0[idx_px] +
                      A0[idx_mx]) * c1
                     - A0[idx] * c0;
      }
    }
  }
}


