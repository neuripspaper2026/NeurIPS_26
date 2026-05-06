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
        int idx_kp1 = idx_center + 1;
        int idx_km1 = idx_center - 1;
        int idx_jp1 = idx_center + nx;
        int idx_jm1 = idx_center - nx;
        int idx_ip1 = idx_center + nx * ny;
        int idx_im1 = idx_center - nx * ny;
        
        Anext[idx_center] = 
          (A0[idx_kp1] +
           A0[idx_km1] +
           A0[idx_jp1] +
           A0[idx_jm1] +
           A0[idx_ip1] +
           A0[idx_im1]) * c1
          - A0[idx_center] * c0;
      }
    }
  }
}


