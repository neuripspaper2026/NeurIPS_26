#include "common.h"

void cpu_stencil(float c0,float c1, float *A0,float * Anext,const int nx, const int ny, const int nz)
{
  const int nx_ny = nx * ny;
  const float c1_mult = c1;
  const float c0_mult = c0;

  for(int i=1;i<nx-1;i++)
  {
    const int i_nx_ny = i * nx_ny;
    for(int j=1;j<ny-1;j++)
    {
      const int base_index = i_nx_ny + j * nx + 1;
      float *Anext_ptr = Anext + base_index;
      float *A0_center = A0 + base_index;
      float *A0_up = A0_center + nx_ny;
      float *A0_down = A0_center - nx_ny;
      float *A0_north = A0_center + nx;
      float *A0_south = A0_center - nx;
      float *A0_east = A0_center + 1;
      float *A0_west = A0_center - 1;

      for(int k=1;k<nz-1;k++)
      {
        *Anext_ptr = ((*A0_up + *A0_down + *A0_north + *A0_south + *A0_east + *A0_west) * c1_mult) - (*A0_center * c0_mult);
        
        Anext_ptr++;
        A0_center++;
        A0_up++;
        A0_down++;
        A0_north++;
        A0_south++;
        A0_east++;
        A0_west++;
      }
    }
  }
}


