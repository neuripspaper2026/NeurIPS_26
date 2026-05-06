#include "common.h"
#ifdef _OPENMP
#include <omp.h>
#endif

void cpu_stencil(float c0,float c1, float *A0,float * Anext,const int nx, const int ny, const int nz)
{
  /* Precompute plane size to avoid repeated multiplications */
  const int nxny = nx * ny;

  int i, j, k;

	/* Parallelize the outer loops with OpenMP, collapse for better work distribution */
#ifdef _OPENMP
#pragma omp parallel for collapse(2) private(i,j,k) schedule(static)
#endif
	for(i = 1; i < nx - 1; i++) {
		for(j = 1; j < ny - 1; j++) {

			/* Compute base index for (i,j,1) to enable reuse */
			int base = i + nx * j + nxny; /* k starts from 1 => +1*nxny */
			const int end_k = nz - 1;

			for(k = 1; k < end_k; k++, base += nxny) {
				const int center = base;
				const int xp = center + 1;
				const int xm = center - 1;
				const int yp = center + nx;
				const int ym = center - nx;
				const int zp = center + nxny;
				const int zm = center - nxny;

				Anext[center] =
					(A0[zp] +
					 A0[zm] +
					 A0[yp] +
					 A0[ym] +
					 A0[xp] +
					 A0[xm]) * c1
					- A0[center] * c0;
			}
		}
	}
}


