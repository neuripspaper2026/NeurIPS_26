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
      const int base_index = i_nx_ny + j * nx;
      float *Anext_row = &Anext[base_index];
      float *A0_center_row = &A0[base_index];
      float *A0_plus1_row = &A0[base_index + nx_ny];
      float *A0_minus1_row = &A0[base_index - nx_ny];
      float *A0_plus_row = &A0[base_index + nx];
      float *A0_minus_row = &A0[base_index - nx];
      
      for(int k=1;k<nz-1;k++)
      {
        Anext_row[k] = 
          (A0_center_row[k + 1] +
           A0_center_row[k - 1] +
           A0_plus_row[k] +
           A0_minus_row[k] +
           A0_plus1_row[k] +
           A0_minus1_row[k]) * c1_mult
          - A0_center_row[k] * c0_mult;
      }
    }
  }
}


