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
      // Compute base indices once per (i,j) pair
      int base_ij = i + nx * j;
      int base_ij_plus_ny = base_ij + nx * ny;
      int base_ij_minus_ny = base_ij - nx * ny;
      int base_i_plus_1_j = (i+1) + nx * j;
      int base_i_minus_1_j = (i-1) + nx * j;
      int base_ij_plus_nx = base_ij + nx;
      int base_ij_minus_nx = base_ij - nx;
      
      for(k=1;k<nz-1;k++)
      {
        int idx = base_ij + nx * ny * k;
        
        Anext[idx] = 
          (A0[idx + nx * ny] +
           A0[idx - nx * ny] +
           A0[idx + nx] +
           A0[idx - nx] +
           A0[idx + 1] +
           A0[idx - 1]) * c1
          - A0[idx] * c0;
      }
    }
  }
}


