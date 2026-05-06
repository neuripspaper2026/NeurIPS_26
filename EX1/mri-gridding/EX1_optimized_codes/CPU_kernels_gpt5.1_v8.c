#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "UDTypes.h"

#define max(x,y) ((x<y)?y:x)
#define min(x,y) ((x>y)?y:x)

#define PI 3.14159265359

static inline float kernel_value_CPU(float v){

  const float z = v*v;

  /* Polynomial evaluation using Horner's rule (unchanged mathematically) */
  const float num =
    z * (z * (z * (z * (z * (z * (z * (z * (z * (z * (z * (z * (z *
    (z * 0.210580722890567e-22f  + 0.380715242345326e-19f ) +
     0.479440257548300e-16f) + 0.435125971262668e-13f ) +
     0.300931127112960e-10f) + 0.160224679395361e-7f  ) +
     0.654858370096785e-5f)  + 0.202591084143397e-2f  ) +
     0.463076284721000e0f)   + 0.754337328948189e2f   ) +
     0.830792541809429e4f)   + 0.571661130563785e6f   ) +
     0.216415572361227e8f)   + 0.356644482244025e9f   ) +
     0.144048298227235e10f);

  const float den = z * (z * (z - 0.307646912682801e4f) +
                         0.347626332405882e7f) -
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

    if (!(*LUT)) {
      *sizeLUT = 0;
      return;
    }

    const float invSize = 1.0f / (float)size;
    unsigned int k;
    for(k=0; k<size; ++k){
      /* compute value to evaluate kernel at
         v in the range 0:(_width/2)^2 */
      v = ((float)k * invSize) * cutoff2;

      /* compute kernel value and store */
      (*LUT)[k] = kernel_value_CPU(beta*sqrtf(1.0f-(v/cutoff2)));
    }
    (*sizeLUT) = size;
  }
}

static inline float kernel_value_LUT(float v, const float* LUT, unsigned int sizeLUT, float _1overCutoff2)
{
  /* v is in [0, cutoff2]; scale to [0, sizeLUT] */
  v *= (float)sizeLUT;
  unsigned int k0 = (unsigned int)(v * _1overCutoff2);

  if (k0 >= sizeLUT - 1U) {
    k0 = sizeLUT - 2U;
  }

  const float v0 = (float)k0 / _1overCutoff2;
  return  LUT[k0] + ((v - v0) * (LUT[k0+1] - LUT[k0]) * _1overCutoff2);
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
  const float kw = (float)params.kernelWidth;
  const float tmp = oversample - 0.5f;
  const float beta = PI * sqrtf(4.0f*kw*kw/(oversample*oversample) * tmp*tmp - 0.8f);

  const int useLUT = params.useLUT;

  unsigned int i;
  for (i=0U; i < n; i++){
    const ReconstructionSample pt = sample[i];

    const float kx = pt.kX;
    const float ky = pt.kY;
    const float kz = pt.kZ;

    if((pt.real == 0.0f && pt.imag == 0.0f) || pt.sdc == 0.0f)
      continue;

    /* compute bounds once */
    const float kx_minus = kx - cutoff;
    const float kx_plus  = kx + cutoff;
    const float ky_minus = ky - cutoff;
    const float ky_plus  = ky + cutoff;
    const float kz_minus = kz - cutoff;
    const float kz_plus  = kz + cutoff;

    NxL = (unsigned int)max(kx_minus, 0.0f);
    NxH = (unsigned int)min(kx_plus,  (float)(size_x-1U));

    NyL = (unsigned int)max(ky_minus, 0.0f);
    NyH = (unsigned int)min(ky_plus,  (float)(size_y-1U));

    NzL = (unsigned int)max(kz_minus, 0.0f);
    NzH = (unsigned int)min(kz_plus,  (float)(size_z-1U));

    /* precompute squared distances */
    for(dz2 = Dz2, nz=(int)NzL; (unsigned int)nz<=NzH; ++nz, ++dz2)
    {
      const float dz = kz - (float)nz;
      *dz2 = dz*dz;
    }
    for(dx2=Dx2,nx=(int)NxL; (unsigned int)nx<=NxH; ++nx,++dx2)
    {
      const float dx = kx - (float)nx;
      *dx2 = dx*dx;
    }
    for(dy2=Dy2, ny=(int)NyL; (unsigned int)ny<=NyH; ++ny,++dy2)
    {
      const float dy = ky - (float)ny;
      *dy2 = dy*dy;
    }

    idxZ = (NzL-1U)*size_x*size_y;
    for(dz2=Dz2, nz=(int)NzL; (unsigned int)nz<=NzH; ++nz, ++dz2)
    {
      /* linear offset into 3-D matrix to get to zposition */
      idxZ += size_x*size_y;

      idxY = (NyL-1U)*size_x;

      /* loop over x indexes, but only if current distance is close enough
         (distance will increase by adding x&y distance) */
      if((*dz2)<cutoff2)
      {
        const float dz2_val = *dz2;
        for(dy2=Dy2, ny=(int)NyL; (unsigned int)ny<=NyH; ++ny, ++dy2)
        {
          /* linear offset IN ADDITION to idxZ to get to Y position */
          idxY += size_x;

          dy2dz2 = dz2_val + (*dy2);

          idx0 = idxY + idxZ;

          /* loop over y indexes, but only if current distance is close enough
             (distance will increase by adding y distance) */
          if(dy2dz2<cutoff2)
          {
            unsigned int idx_local = idx0;
            for(dx2=Dx2, nx=(int)NxL; (unsigned int)nx<=NxH; ++nx, ++dx2)
            {
              /* value to evaluate kernel at */
              v = dy2dz2+(*dx2);

              if(v<cutoff2)
              {
                /* linear index of (x,y,z) point */
                /* idx_local already holds idx0 + nx */
                idx = idx_local + (unsigned int)nx;

                /* kernel weighting value */
                if (useLUT){
                  w = kernel_value_LUT(v, LUT, sizeLUT, _1overCutoff2) * pt.sdc;
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

  return 0;
}
