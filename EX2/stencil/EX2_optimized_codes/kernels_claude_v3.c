#include "common.h"
#ifdef _OPENMP
#include <omp.h>
#endif

void cpu_stencil(float c0,float c1, float *A0,float * Anext,const int nx, const int ny, const int nz)
{
  int i, j, k;
  
  #ifdef _OPENMP
  #pragma omp parallel for collapse(2) private(i, j, k) schedule(static)
  #endif
  for(i=1;i<nx-1;i++)
  {
    for(j=1;j<ny-1;j++)
    {
      for(k=1;k<nz-1;k++)
      {
        const int idx_center = i + nx * (j + ny * k);
        const int idx_k_plus  = idx_center + nx * ny;
        const int idx_k_minus = idx_center - nx * ny;
        const int idx_j_plus  = idx_center + nx;
        const int idx_j_minus = idx_center - nx;
        const int idx_i_plus  = idx_center + 1;
        const int idx_i_minus = idx_center - 1;
        
        Anext[idx_center] = 
          (A0[idx_k_plus] +
           A0[idx_k_minus] +
           A0[idx_j_plus] +
           A0[idx_j_minus] +
           A0[idx_i_plus] +
           A0[idx_i_minus]) * c1
          - A0[idx_center] * c0;
      }
    }
  }
}


