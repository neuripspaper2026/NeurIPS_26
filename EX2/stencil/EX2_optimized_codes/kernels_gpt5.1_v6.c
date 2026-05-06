#include "common.h"
#ifdef _OPENMP
#include <omp.h>
#endif

void cpu_stencil(float c0,float c1, float *A0,float * Anext,const int nx, const int ny, const int nz)
{

  int i, j, k;

  /* Precompute plane stride for better aliasing information */
  const int nxny = nx * ny;

#ifdef _OPENMP
  /* Parallelize outer loops; collapse for better load balance and locality */
#pragma omp parallel for collapse(2) private(i,j,k) schedule(static)
#endif
	for(i=1;i<nx-1;i++)
	{
		for(j=1;j<ny-1;j++)
		{
      /* Precompute base index for this (i,j) column */
      const int base = i + nx * j;
			for(k=1;k<nz-1;k++)
			{
        const int idx   = base + nxny * k;
        const int idx_zp = idx + nxny;
        const int idx_zm = idx - nxny;
        const int idx_yp = idx + nx;
        const int idx_ym = idx - nx;
        const int idx_xp = idx + 1;
        const int idx_xm = idx - 1;

				Anext[idx] = 
				(A0[idx_zp] +
				A0[idx_zm] +
				A0[idx_yp] +
				A0[idx_ym] +
				A0[idx_xp] +
				A0[idx_xm])*c1
				- A0[idx]*c0;
			}
		}
	}

}


