#include "common.h"

void cpu_stencil(float c0,float c1, float *A0,float * Anext,const int nx, const int ny, const int nz)
{
  const int nx_ny = nx * ny;
  const float c1_mult = c1;
  const float c0_mult = c0;

  for(int i=1; i<nx-1; i++)
  {
    const int i_nx_ny = i * nx_ny;
    for(int j=1; j<ny-1; j++)
    {
      const int j_nx = j * nx;
      const int base_index = i_nx_ny + j_nx + 1;
      const int end_index = base_index + (nz - 2);
      
      for(int k=base_index; k<end_index; k++)
      {
        const int center = k;
        const int east = k + 1;
        const int west = k - 1;
        const int north = k + nx;
        const int south = k - nx;
        const int top = k + nx_ny;
        const int bottom = k - nx_ny;
        
        Anext[center] = 
          (A0[east] + A0[west] + A0[north] + A0[south] + A0[top] + A0[bottom]) * c1_mult
          - A0[center] * c0_mult;
      }
    }
  }
}


