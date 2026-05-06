#include "common.h"

void cpu_stencil(float c0,float c1, float *A0,float * Anext,const int nx, const int ny, const int nz)
{
  const int nx_ny = nx * ny;
  const float c1_A0_k_plus_1 = c1;
  const float c1_A0_k_minus_1 = c1;
  const float c1_A0_j_plus_1 = c1;
  const float c1_A0_j_minus_1 = c1;
  const float c1_A0_i_plus_1 = c1;
  const float c1_A0_i_minus_1 = c1;
  const float c0_A0 = c0;

  int i, j, k;
  for(i=1;i<nx-1;i++)
  {
    const int i_nx = i * nx;
    for(j=1;j<ny-1;j++)
    {
      const int base_index = i_nx + j * nx;
      float *Anext_ptr = Anext + base_index;
      float *A0_ptr = A0 + base_index;
      
      for(k=1;k<nz-1;k++)
      {
        const int k_plus_1 = k + 1;
        const int k_minus_1 = k - 1;
        const int j_plus_1_nx = (j + 1) * nx;
        const int j_minus_1_nx = (j - 1) * nx;
        const int i_plus_1_nx = (i + 1) * nx;
        const int i_minus_1_nx = (i - 1) * nx;
        
        *(Anext_ptr + k) = 
          (*(A0_ptr + k_plus_1) +
           *(A0_ptr + k_minus_1) +
           *(A0 + base_index - nx + k) +
           *(A0 + base_index + nx + k) +
           *(A0 + base_index + nx_ny + k) +
           *(A0 + base_index - nx_ny + k)) * c1_A0_k_plus_1
          - *(A0_ptr + k) * c0_A0;
      }
    }
  }
}


