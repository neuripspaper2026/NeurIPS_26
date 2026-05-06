#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "UDTypes.h"

#define max(x,y) ((x<y)?y:x)
#define min(x,y) ((x>y)?y:x)

#define PI 3.14159265359

float kernel_value_CPU(float v){

  /* Use double for intermediate computations for better precision and
     reduced rounding accumulation in long polynomial chains, then cast
     back to float. */
  const double z = (double)v * (double)v;

  const double num =
    z * ( z * ( z * ( z * ( z * ( z * ( z * ( z * ( z * ( z * ( z * ( z * ( z *
    ( z * 0.210580722890567e-22  + 0.380715242345326e-19 ) +
       0.479440257548300e-16 ) + 0.435125971262668e-13 ) +
       0.300931127112960e-10 ) + 0.160224679395361e-7  ) +
       0.654858370096785e-5  ) + 0.202591084143397e-2  ) +
       0.463076284721000e0   ) + 0.754337328948189e2   ) +
       0.830792541809429e4   ) + 0.571661130563785e6   ) +
       0.216415572361227e8   ) + 0.356644482244025e9   ) +
       0.144048298227235e10 );

  const double den =
    z * ( z * ( z - 0.307646912682801e4 ) + 0.347626332405882e7 ) -
    0.144048298227235e10;

  return (float)(-num / den);
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

    const float invSize = 1.0f/(float)size;
    unsigned int k;
    for(k=0; k<size; ++k){
      /* v in the range 0:(_width/2)^2 */
      v = ((float)k)*invSize*cutoff2;

      /* compute kernel value and store */
      (*LUT)[k] = kernel_value_CPU(beta*sqrtf(1.0f-(v/cutoff2)));
    }
    (*sizeLUT) = size;
  }
}

float kernel_value_LUT(float v, float* LUT, int sizeLUT, float _1overCutoff2)
{
  /* Precompute frequently used values */
  const float scaled = v * (float)sizeLUT;
  const float kf = scaled * _1overCutoff2;
  const unsigned int k0 = (unsigned int)kf;
  const float v0 = (float)k0 / _1overCutoff2;

  return LUT[k0] + ((scaled - v0) * (LUT[k0+1] - LUT[k0]) * (1.0f/_1overCutoff2));
}

int gridding_Gold(unsigned int n, parameters params, ReconstructionSample* sample, float* LUT, unsigned int sizeLUT, cmplx* gridData, float* sampleDensity){

  unsigned int NxL, NxH;
  unsigned int NyL, NyH;
  unsigned int NzL, NzH;

  int nx;
  int ny;
  int nz;

  float w;
  unsigned int idx;
  unsigned int idx0;

  unsigned int idxZ;
  unsigned int idxY;

  float Dx2[100];
  float Dy2[100];
  float Dz2[100];
  float *dx2=NULL;
  float *dy2=NULL;
  float *dz2=NULL;

  float dy2dz2;
  float v;

  const unsigned int size_x = (unsigned int)params.gridSize[0];
  const unsigned int size_y = (unsigned int)params.gridSize[1];
  const unsigned int size_z = (unsigned int)params.gridSize[2];

  const float cutoff = ((float)(params.kernelWidth))*0.5f;   /* cutoff radius */
  const float cutoff2 = cutoff*cutoff;                       /* square of cutoff radius */
  const float _1overCutoff2 = 1.0f/cutoff2;                  /* 1 over square of cutoff radius */

  const float oversample = params.oversample;
  const float kernelWidth = (float)params.kernelWidth;
  const float tmp = (oversample - 0.5f);
  const float beta = (float)(PI * sqrt( (4.0f*kernelWidth*kernelWidth/(oversample*oversample)) * (tmp*tmp) - 0.8f ));

  const int useLUT = params.useLUT;

  /* Precompute 1.0f/_1overCutoff2 for kernel_value_LUT */
  const float inv_1overCutoff2 = 1.0f / _1overCutoff2;

  unsigned int i;
  for (i=0; i < n; i++){
    const ReconstructionSample pt = sample[i];

    const float kx = pt.kX;
    const float ky = pt.kY;
    const float kz = pt.kZ;

    NxL = (unsigned int)max(kx - cutoff, 0.0f);
    NxH = (unsigned int)min(kx + cutoff, (float)(size_x-1U));

    NyL = (unsigned int)max(ky - cutoff, 0.0f);
    NyH = (unsigned int)min(ky + cutoff, (float)(size_y-1U));

    NzL = (unsigned int)max(kz - cutoff, 0.0f);
    NzH = (unsigned int)min(kz + cutoff, (float)(size_z-1U));

    if((pt.real != 0.0f || pt.imag != 0.0f) && pt.sdc!=0.0f)
    {
      const float kz_local = kz;
      const float kx_local = kx;
      const float ky_local = ky;

      /* Precompute distance in z */
      dz2 = Dz2;
      for(nz=(int)NzL; (unsigned int)nz<=NzH; ++nz, ++dz2)
      {
        const float dz = kz_local - (float)nz;
        *dz2 = dz*dz;
      }

      /* Precompute distance in x */
      dx2 = Dx2;
      for(nx=(int)NxL; (unsigned int)nx<=NxH; ++nx, ++dx2)
      {
        const float dx = kx_local - (float)nx;
        *dx2 = dx*dx;
      }

      /* Precompute distance in y */
      dy2 = Dy2;
      for(ny=(int)NyL; (unsigned int)ny<=NyH; ++ny, ++dy2)
      {
        const float dy = ky_local - (float)ny;
        *dy2 = dy*dy;
      }

      idxZ = (NzL-1U)*size_x*size_y;
      dz2 = Dz2;
      for(nz=(int)NzL; (unsigned int)nz<=NzH; ++nz, ++dz2)
      {
        /* linear offset into 3-D matrix to get to zposition */
        idxZ += size_x*size_y;

        idxY = (NyL-1U)*size_x;

        const float dz2_val = *dz2;

        /* loop over x indexes, but only if current distance is close enough (distance will increase by adding x&y distance) */
        if(dz2_val<cutoff2)
        {
          dy2 = Dy2;
          for(ny=(int)NyL; (unsigned int)ny<=NyH; ++ny, ++dy2)
          {
            /* linear offset IN ADDITION to idxZ to get to Y position */
            idxY += size_x;

            dy2dz2 = dz2_val + (*dy2);

            idx0 = idxY + idxZ;

            /* loop over y indexes, but only if current distance is close enough (distance will increase by adding y distance) */
            if(dy2dz2<cutoff2)
            {
              dx2 = Dx2;
              for(nx=(int)NxL; (unsigned int)nx<=NxH; ++nx, ++dx2)
              {
                /* value to evaluate kernel at */
                v = dy2dz2 + (*dx2);

                if(v<cutoff2)
                {
                  /* linear index of (x,y,z) point */
                  idx = (unsigned int)nx + idx0;

                  /* kernel weighting value */
                  if (useLUT){
                    /* inline and reuse precomputed inverse */
                    const float scaled = v * (float)sizeLUT;
                    const float kf = scaled * _1overCutoff2;
                    const unsigned int k0 = (unsigned int)kf;
                    const float v0 = (float)k0 * inv_1overCutoff2;
                    const float lutw = LUT[k0] + ((scaled - v0) * (LUT[k0+1] - LUT[k0]) * inv_1overCutoff2);
                    w = lutw * pt.sdc;
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
  }

  return 0;
}
