#include "common.h"
#ifdef _OPENMP
#include <omp.h>
#endif

void cpu_stencil(float c0,float c1, float *A0,float * Anext,const int nx, const int ny, const int nz)
{
  int i, j, k;
  
  #ifdef _OPENMP
  #pragma omp parallel for collapse(2) private(i, j, k) schedule(static)
  #endif
  for(i=1;i<nx-1;i++)
  {
    for(j=1;j<ny-1;j++)
    {
      for(k=1;k<nz-1;k++)
      {
        int idx_center = Index3D(nx, ny, i, j, k);
        int idx_kp = idx_center + 1;
        int idx_km = idx_center - 1;
        int idx_jp = idx_center + nx;
        int idx_jm = idx_center - nx;
        int idx_ip = idx_center + nx * ny;
        int idx_im = idx_center - nx * ny;
        
        Anext[idx_center] = 
          (A0[idx_kp] +
           A0[idx_km] +
           A0[idx_jp] +
           A0[idx_jm] +
           A0[idx_ip] +
           A0[idx_im]) * c1
          - A0[idx_center] * c0;
      }
    }
  }
}


