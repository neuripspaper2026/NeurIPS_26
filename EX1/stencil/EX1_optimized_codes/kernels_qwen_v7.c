#include "common.h"

void cpu_stencil(float c0,float c1, float *A0,float * Anext,const int nx, const int ny, const int nz)
{
  const int nx_ny = nx * ny;
  const int nx_ny_nz = nx_ny * nz;
  
  for(int i=1;i<nx-1;i++)
  {
    const int i_nx = i * nx;
    for(int j=1;j<ny-1;j++)
    {
      const int base_index = i_nx + j * nx_ny;
      float *Anext_ptr = Anext + base_index;
      float *A0_center_ptr = A0 + base_index;
      float *A0_up_ptr = A0_center_ptr + nx_ny;
      float *A0_down_ptr = A0_center_ptr - nx_ny;
      float *A0_front_ptr = A0_center_ptr + nx;
      float *A0_back_ptr = A0_center_ptr - nx;
      float *A0_left_ptr = A0_center_ptr + 1;
      float *A0_right_ptr = A0_center_ptr - 1;
      
      for(int k=1;k<nz-1;k++)
      {
        *Anext_ptr = (
          *(A0_up_ptr) +
          *(A0_down_ptr) +
          *(A0_front_ptr) +
          *(A0_back_ptr) +
          *(A0_left_ptr) +
          *(A0_right_ptr)
        ) * c1 - *(A0_center_ptr) * c0;
        
        Anext_ptr++;
        A0_center_ptr++;
        A0_up_ptr++;
        A0_down_ptr++;
        A0_front_ptr++;
        A0_back_ptr++;
        A0_left_ptr++;
        A0_right_ptr++;
      }
    }
  }
}


