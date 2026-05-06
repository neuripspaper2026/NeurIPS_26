#include "common.h"

void cpu_stencil(float c0,float c1, float *A0,float * Anext,const int nx, const int ny, const int nz)
{
  int i, j, k;
  const int nx_ny = nx * ny;
  
  for(k=1;k<nz-1;k++)
  {
    const int k_offset = nx_ny * k;
    const int k_plus_offset = nx_ny * (k + 1);
    const int k_minus_offset = nx_ny * (k - 1);
    
    for(j=1;j<ny-1;j++)
    {
      const int j_offset = nx * j;
      const int j_plus_offset = nx * (j + 1);
      const int j_minus_offset = nx * (j - 1);
      
      const int base_idx = j_offset + k_offset;
      const int j_plus_idx = j_plus_offset + k_offset;
      const int j_minus_idx = j_minus_offset + k_offset;
      const int k_plus_idx = j_offset + k_plus_offset;
      const int k_minus_idx = j_offset + k_minus_offset;
      
      for(i=1;i<nx-1;i++)
      {
        const int center = i + base_idx;
        Anext[center] = 
          (A0[i + k_plus_idx] +
           A0[i + k_minus_idx] +
           A0[i + j_plus_idx] +
           A0[i + j_minus_idx] +
           A0[(i + 1) + base_idx] +
           A0[(i - 1) + base_idx]) * c1
          - A0[center] * c0;
      }
    }
  }
}


