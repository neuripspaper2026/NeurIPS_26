#include "common.h"

void cpu_stencil(float c0,float c1, float *A0,float * Anext,const int nx, const int ny, const int nz)
{
  int i, j, k;
  const int nxny = nx * ny;
  
  for(k=1;k<nz-1;k++)
  {
    const int k_offset = nxny * k;
    const int k_plus_offset = nxny * (k + 1);
    const int k_minus_offset = nxny * (k - 1);
    
    for(j=1;j<ny-1;j++)
    {
      const int j_offset = nx * j;
      const int j_plus_offset = nx * (j + 1);
      const int j_minus_offset = nx * (j - 1);
      
      const int base_jk = j_offset + k_offset;
      const int base_jk_plus = j_plus_offset + k_offset;
      const int base_jk_minus = j_minus_offset + k_offset;
      const int base_j_kplus = j_offset + k_plus_offset;
      const int base_j_kminus = j_offset + k_minus_offset;
      
      for(i=1;i<nx-1;i++)
      {
        const int idx = i + base_jk;
        
        Anext[idx] = 
          (A0[i + base_j_kplus] +
           A0[i + base_j_kminus] +
           A0[i + base_jk_plus] +
           A0[i + base_jk_minus] +
           A0[i + 1 + base_jk] +
           A0[i - 1 + base_jk])*c1
          - A0[idx]*c0;
      }
    }
  }
}


