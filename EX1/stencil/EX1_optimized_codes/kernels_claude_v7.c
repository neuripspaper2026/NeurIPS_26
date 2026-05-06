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
      float *Anext_ptr = Anext + base_ij + nxny;
      const float *A0_ptr = A0 + base_ij + nxny;
      
      for(k=1;k<nz-1;k++)
      {
        float center = A0_ptr[0];
        float sum = A0_ptr[nxny] + A0_ptr[-nxny] +
                    A0_ptr[nx] + A0_ptr[-nx] +
                    A0_ptr[1] + A0_ptr[-1];
        
        *Anext_ptr = sum * c1 - center * c0;
        
        Anext_ptr += nxny;
        A0_ptr += nxny;
      }
    }
  }
}


