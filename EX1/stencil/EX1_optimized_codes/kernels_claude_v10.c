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
      float *Anext_row = &Anext[base_ij + nxny];
      const float *A0_row = &A0[base_ij + nxny];
      
      for(k=1;k<nz-1;k++)
      {
        float center = A0_row[0];
        float sum = A0_row[nxny] + A0_row[-nxny] +
                    A0_row[nx] + A0_row[-nx] +
                    A0_row[1] + A0_row[-1];
        
        Anext_row[0] = sum * c1 - center * c0;
        
        Anext_row += nxny;
        A0_row += nxny;
      }
    }
  }
}


