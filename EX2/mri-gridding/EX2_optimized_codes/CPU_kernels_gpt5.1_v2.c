#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "UDTypes.h"

#define max(x,y) ((x<y)?y:x)
#define min(x,y) ((x>y)?y:x)

#define PI 3.14159265359

static inline float kernel_value_CPU(float v) {

  const float z = v * v;

  /* Horner form of polynomial */
  float num =
    z * (z * (z * (z * (z * (z * (z * (z * (z * (z * (z * (z * (z *
    (z * 0.210580722890567e-22f  + 0.380715242345326e-19f ) +
     0.479440257548300e-16f) + 0.435125971262668e-13f ) +
     0.300931127112960e-10f) + 0.160224679395361e-7f  ) +
     0.654858370096785e-5f)  + 0.202591084143397e-2f  ) +
     0.463076284721000e0f)   + 0.754337328948189e2f   ) +
     0.830792541809429e4f)   + 0.571661130563785e6f   ) +
     0.216415572361227e8f)   + 0.356644482244025e9f   ) +
     0.144048298227235e10f);

  float den = z * (z * (z - 0.307646912682801e4f) + 0.347626332405882e7f) -
              0.144048298227235e10f;

  return -num / den;
}

void calculateLUT(float beta, float width, float** LUT, unsigned int* sizeLUT){
  float v;
  const float cutoff2 = (width*width)*0.25f;

  unsigned int size;

  if(width > 0.0f){
    /* compute size of LUT based on kernel width */
    size = (unsigned int)(10000.0f*width);

    /* allocate memory */
    (*LUT) = (float*) malloc (size*sizeof(float));

    const float invSize = 1.0f / (float)size;
    unsigned int k;
    for(k=0; k < size; ++k){
      /* compute value to evaluate kernel at
       * v in the range 0:(_width/2)^2
       */
      v = ((float)k) * invSize * cutoff2;

      /* compute kernel value and store */
      (*LUT)[k] = kernel_value_CPU(beta*sqrtf(1.0f-(v/cutoff2)));
    }
    (*sizeLUT) = size;
  }
}

static inline float kernel_value_LUT(float v, const float* __restrict LUT,
                                     int sizeLUT, float _1overCutoff2)
{
  /* v is already squared distance; scale and lookup with linear interpolation */
  const float scaled = v * (float)sizeLUT;
  const unsigned int k0 = (unsigned int)(scaled * _1overCutoff2);
  const float v0 = ((float)k0) / _1overCutoff2;
  const float dv = scaled - v0;
  const float diff = LUT[k0 + 1U] - LUT[k0];
  return LUT[k0] + (dv * diff / _1overCutoff2);
}

int gridding_Gold(unsigned int n, parameters params,
                  ReconstructionSample* __restrict sample,
                  float* __restrict LUT, unsigned int sizeLUT,
                  cmplx* __restrict gridData,
                  float* __restrict sampleDensity)
{

  const unsigned int size_x = (unsigned int)params.gridSize[0];
  const unsigned int size_y = (unsigned int)params.gridSize[1];
  const unsigned int size_z = (unsigned int)params.gridSize[2];

  const float cutoff  = ((float)(params.kernelWidth))*0.5f; /* cutoff radius */
  const float cutoff2 = cutoff*cutoff;                      /* square of cutoff radius */
  const float _1overCutoff2 = 1.0f/cutoff2;                 /* 1 over square of cutoff radius */

  const float oversample = params.oversample;
  const float kw = (float)params.kernelWidth;
  const float beta = PI * sqrtf(4.0f*kw*kw/(oversample*oversample) *
                                (oversample-0.5f)*(oversample-0.5f)-0.8f);

  /* Thread-local scratch buffers to avoid races and heap allocations */
#pragma omp parallel
  {
    float Dx2[100];
    float Dy2[100];
    float Dz2[100];

#pragma omp for schedule(static)
    for (int i = 0; i < (int)n; i++){
      ReconstructionSample pt = sample[i];

      const float kx = pt.kX;
      const float ky = pt.kY;
      const float kz = pt.kZ;

      unsigned int NxL = (unsigned int)max(kx - cutoff, 0.0f);
      unsigned int NxH = (unsigned int)min(kx + cutoff, (float)(size_x - 1U));

      unsigned int NyL = (unsigned int)max(ky - cutoff, 0.0f);
      unsigned int NyH = (unsigned int)min(ky + cutoff, (float)(size_y - 1U));

      unsigned int NzL = (unsigned int)max(kz - cutoff, 0.0f);
      unsigned int NzH = (unsigned int)min(kz + cutoff, (float)(size_z - 1U));

      if ((pt.real != 0.0f || pt.imag != 0.0f) && pt.sdc != 0.0f)
      {
        int nz;
        int nx;
        int ny;

        float *dx2;
        float *dy2;
        float *dz2;

        /* precompute squared distances in each dimension */
        for(dz2 = Dz2, nz = (int)NzL; nz <= (int)NzH; ++nz, ++dz2)
        {
          const float dz = kz - (float)nz;
          *dz2 = dz * dz;
        }
        for(dx2 = Dx2, nx = (int)NxL; nx <= (int)NxH; ++nx, ++dx2)
        {
          const float dx = kx - (float)nx;
          *dx2 = dx * dx;
        }
        for(dy2 = Dy2, ny = (int)NyL; ny <= (int)NyH; ++ny, ++dy2)
        {
          const float dy = ky - (float)ny;
          *dy2 = dy * dy;
        }

        unsigned int idxZ = (NzL ? (NzL - 1U) : 0U) * size_x * size_y;
        for(dz2 = Dz2, nz = (int)NzL; nz <= (int)NzH; ++nz, ++dz2)
        {
          idxZ += size_x * size_y; /* linear offset into 3-D matrix to get to zposition */

          unsigned int idxY = (NyL ? (NyL - 1U) : 0U) * size_x;

          /* loop over y indexes, but only if current distance is close enough
           * (distance will increase by adding x&y distance)
           */
          if((*dz2) < cutoff2)
          {
            for(dy2 = Dy2, ny = (int)NyL; ny <= (int)NyH; ++ny, ++dy2)
            {
              idxY += size_x;   /* linear offset IN ADDITION to idxZ to get to Y position */

              float dy2dz2 = (*dz2) + (*dy2);

              const unsigned int idx0 = idxY + idxZ;

              /* loop over x indexes, but only if current distance is close enough
               * (distance will increase by adding y distance)
               */
              if(dy2dz2 < cutoff2)
              {
                for(dx2 = Dx2, nx = (int)NxL; nx <= (int)NxH; ++nx, ++dx2)
                {
                  /* value to evaluate kernel at */
                  const float v = dy2dz2 + (*dx2);

                  if(v < cutoff2)
                  {
                    const unsigned int idx = (unsigned int)nx + idx0;

                    /* kernel weighting value */
                    float w;
                    if (params.useLUT){
                      w = kernel_value_LUT(v, LUT, (int)sizeLUT, _1overCutoff2) * pt.sdc;
                    } else {
                      w = kernel_value_CPU(beta*sqrtf(1.0f-(v*_1overCutoff2))) * pt.sdc;
                    }

                    /* grid data */
                    gridData[idx].real += (w*pt.real);
                    gridData[idx].imag += (w*pt.imag);

                    /* estimate sample density */
                    sampleDensity[idx] += 1.0f;
                  }
                }
              }
            }
          }
        }
      }
    } /* end for i */
  } /* end parallel */

  return 0;
}
