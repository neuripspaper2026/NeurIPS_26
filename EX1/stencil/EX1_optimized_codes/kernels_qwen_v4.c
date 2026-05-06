#include "common.h"

void cpu_stencil(float c0,float c1, float *A0,float * Anext,const int nx, const int ny, const int nz)
{
  const int nx_ny = nx * ny;
  const int nx_ny_nz = nx_ny * nz;
  
  for(int i = nx_ny + nx + 1; i < nx_ny_nz - nx_ny - nx - 1; i++) {
    if ((i % nx) == 0 || (i % nx) == nx - 1) continue;
    if ((i / nx) % ny == 0 || (i / nx) % ny == ny - 1) continue;
    
    Anext[i] = (A0[i + 1] + A0[i - 1] + 
                A0[i + nx] + A0[i - nx] + 
                A0[i + nx_ny] + A0[i - nx_ny]) * c1 - 
               A0[i] * c0;
  }
}


