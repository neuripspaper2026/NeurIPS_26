#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "UDTypes.h"

#define max(x,y) ((x<y)?y:x)
#define min(x,y) ((x>y)?y:x)

#define PI 3.14159265359

/* Evaluate kernel value on CPU using precomputed polynomial approximation. */
static inline float kernel_value_CPU(float v){

  const float z = v*v;

  /* polynomials taken from
   * http://ccrma.stanford.edu/CCRMA/Courses/422/projects/kbd/kbdwindow.cpp
   */
  const float num =
    z*(z*(z*(z*(z*(z*(z*(z*(z*(z*(z*(z*(z*
      0.210580722890567e-22f  + 0.380715242345326e-19f ) +
      0.479440257548300e-16f) + 0.435125971262668e-13f ) +
      0.300931127112960e-10f) + 0.160224679395361e-7f  ) +
      0.654858370096785e-5f)  + 0.202591084143397e-2f  ) +
      0.463076284721000e0f)   + 0.754337328948189e2f   ) +
      0.830792541809429e4f)   + 0.571661130563785e6f   ) +
      0.216415572361227e8f)   + 0.356644482244025e9f   ) +
      0.144048298227235e10f);

  const float den =
    z*(z*(z-0.307646912682801e4f)+0.347626332405882e7f)-0.144048298227235e10f;

  return -num/den;
}

void calculateLUT(float beta, float width, float** LUT, unsigned int* sizeLUT){
  const float cutoff2 = (width*width)*0.25f;

  if(width > 0.0f){
    /* compute size of LUT based on kernel width */
    const unsigned int size = (unsigned int)(10000.0f*width);

    /* allocate memory */
    *LUT = (float*) malloc (size*sizeof(float));

    const float invSize = 1.0f/(float)size;
    unsigned int k;
    for(k=0; k<size; ++k){
      /* v in the range 0:(_width/2)^2 */
      const float v = ((float)k)*invSize*cutoff2;

      /* compute kernel value and store */
      (*LUT)[k] = kernel_value_CPU(beta*sqrtf(1.0f-(v/cutoff2)));
    }
    *sizeLUT = size;
  }
}

static inline float kernel_value_LUT(float v, const float* LUT, int sizeLUT, float _1overCutoff2)
{
  /* scale v into LUT domain: v' = v * sizeLUT */
  v *= (float)sizeLUT;
  const float scaled = v * _1overCutoff2;
  const unsigned int k0 = (unsigned int)scaled;
  const float v0 = (float)k0 / _1overCutoff2;
  const float diff = LUT[k0+1] - LUT[k0];
  return  LUT[k0] + ((v-v0)*diff*_1overCutoff2);
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

  const float cutoff  = ((float)(params.kernelWidth))*0.5f;  /* cutoff radius */
  const float cutoff2 = cutoff*cutoff;                       /* square of cutoff radius */
  const float _1overCutoff2 = 1.0f/cutoff2;                  /* 1 over square of cutoff radius */

  const float kw = (float)params.kernelWidth;
  const float over = params.oversample;
  const float tmp = (4.0f*kw*kw)/(over*over) * (over-0.5f)*(over-0.5f)-0.8f;
  const float beta = PI * sqrtf(tmp);

  const int useLUT = params.useLUT;

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

      for(dz2 = Dz2, nz=(int)NzL; (unsigned int)nz<=NzH; ++nz, ++dz2)
      {
        const float diff = kz_local-(float)nz;
        *dz2 = diff*diff;
      }
      for(dx2=Dx2,nx=(int)NxL; (unsigned int)nx<=NxH; ++nx,++dx2)
      {
        const float diff = kx_local-(float)nx;
        *dx2 = diff*diff;
      }
      for(dy2=Dy2, ny=(int)NyL; (unsigned int)ny<=NyH; ++ny,++dy2)
      {
        const float diff = ky_local-(float)ny;
        *dy2 = diff*diff;
      }

      const unsigned int sliceStride = size_x*size_y;
      const unsigned int rowStride   = size_x;

      idxZ = (NzL-1U)*sliceStride;
      for(dz2=Dz2, nz=(int)NzL; (unsigned int)nz<=NzH; ++nz, ++dz2)
      {
        /* linear offset into 3-D matrix to get to zposition */
        idxZ += sliceStride;

        idxY = (NyL-1U)*rowStride;

        /* loop over y indexes, but only if current distance is close enough */
        if((*dz2)<cutoff2)
        {
          for(dy2=Dy2, ny=(int)NyL; (unsigned int)ny<=NyH; ++ny, ++dy2)
          {
            /* linear offset IN ADDITION to idxZ to get to Y position */
            idxY += rowStride;

            dy2dz2 = (*dz2)+(*dy2);

            idx0 = idxY + idxZ;

            if(dy2dz2<cutoff2)
            {
              for(dx2=Dx2, nx=(int)NxL; (unsigned int)nx<=NxH; ++nx, ++dx2)
              {
                v = dy2dz2+(*dx2);

                if(v<cutoff2)
                {
                  /* linear index of (x,y,z) point */
                  idx = (unsigned int)nx + idx0;

                  /* kernel weighting value */
                  if (useLUT){
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
  }

  return 0;
}
