#include "common.h"

void cpu_stencil(float c0,float c1, float *A0,float * Anext,const int nx, const int ny, const int nz)
{
  int i, j, k;
  const int nxny = nx * ny;
  
  for(i=1;i<nx-1;i++)
  {
    for(j=1;j<ny-1;j++)
    {
      const int base_ij = i + nx * j;
      const int base_ij_prev = base_ij - nx;
      const int base_ij_next = base_ij + nx;
      
      for(k=1;k<nz-1;k++)
      {
        const int idx_center = base_ij + nxny * k;
        const int idx_k_prev = idx_center - nxny;
        const int idx_k_next = idx_center + nxny;
        
        Anext[idx_center] = 
          (A0[idx_k_next] +
           A0[idx_k_prev] +
           A0[base_ij_next + nxny * k] +
           A0[base_ij_prev + nxny * k] +
           A0[idx_center + 1] +
           A0[idx_center - 1]) * c1
          - A0[idx_center] * c0;
      }
    }
  }
}


