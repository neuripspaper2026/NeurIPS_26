#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "UDTypes.h"

#define max(x,y) ((x<y)?y:x)
#define min(x,y) ((x>y)?y:x)

#define PI 3.14159265359

static inline float kernel_value_CPU(float v){

  const float z = v*v;

  /* polynomials taken from http://ccrma.stanford.edu/CCRMA/Courses/422/projects/kbd/kbdwindow.cpp */
  float num = (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z*
  (z* 0.210580722890567e-22f  + 0.380715242345326e-19f ) +
   0.479440257548300e-16f) + 0.435125971262668e-13f ) +
   0.300931127112960e-10f) + 0.160224679395361e-7f  ) +
   0.654858370096785e-5f)  + 0.202591084143397e-2f  ) +
   0.463076284721000e0f)   + 0.754337328948189e2f   ) +
   0.830792541809429e4f)   + 0.571661130563785e6f   ) +
   0.216415572361227e8f)   + 0.356644482244025e9f   ) +
   0.144048298227235e10f);

  float den = (z*(z*(z-0.307646912682801e4f)+0.347626332405882e7f)-0.144048298227235e10f);

  return -num/den;
}

void calculateLUT(float beta, float width, float** LUT, unsigned int* sizeLUT){
  float v;
  float cutoff2 = (width*width)*0.25f;

  unsigned int size;

  if(width > 0.0f){
    /* compute size of LUT based on kernel width */
    size = (unsigned int)(10000.0f*width);

    /* allocate memory */
    (*LUT) = (float*) malloc (size*sizeof(float));

    const float invSize = 1.0f/(float)size;
    const float invCutoff2 = 1.0f/cutoff2;

    unsigned int k;
    for(k=0; k<size; ++k){
      /* compute value to evaluate kernel at
       * v in the range 0:(_width/2)^2
       */
      v = ((float)k)*invSize*cutoff2;

      /* compute kernel value and store */
      (*LUT)[k] = kernel_value_CPU(beta*sqrtf(1.0f-(v*invCutoff2)));
    }
    (*sizeLUT) = size;
  }
}

static inline float kernel_value_LUT(float v, const float* __restrict LUT, int sizeLUT, float _1overCutoff2)
{
  /* v corresponds to a squared radius in [0,cutoff2]
   * LUT is sampled uniformly over [0,cutoff2] with sizeLUT samples
   */
  const float scaled = v * (float)sizeLUT * _1overCutoff2; /* == v/cutoff2 * sizeLUT */
  unsigned int k0 = (unsigned int)scaled;
  if (k0 >= (unsigned int)(sizeLUT-1)) {
    k0 = (unsigned int)(sizeLUT-2);
  }
  const float v0 = (float)k0 / _1overCutoff2;
  return  LUT[k0] + ((scaled - (v0 * (float)sizeLUT * _1overCutoff2))*(LUT[k0+1]-LUT[k0]));
}

int gridding_Gold(unsigned int n, parameters params, ReconstructionSample* sample, float* LUT, unsigned int sizeLUT, cmplx* gridData, float* sampleDensity){

  unsigned int size_x = (unsigned int)params.gridSize[0];
  unsigned int size_y = (unsigned int)params.gridSize[1];
  unsigned int size_z = (unsigned int)params.gridSize[2];

  const float kernelWidth = params.kernelWidth;
  const float cutoff = kernelWidth*0.5f;          /* cutoff radius */
  const float cutoff2 = cutoff*cutoff;            /* square of cutoff radius */
  const float _1overCutoff2 = 1.0f/cutoff2;       /* 1 over square of cutoff radius */

  const float oversample = params.oversample;
  const float t = oversample - 0.5f;
  const float kernelWidth2 = kernelWidth * kernelWidth;
  const float os2 = oversample * oversample;
  const float beta = PI * sqrtf((4.0f*kernelWidth2/os2) * (t*t) - 0.8f);

  const int useLUT = params.useLUT;

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
      unsigned int NxH = (unsigned int)min(kx + cutoff, (float)(size_x-1));

      unsigned int NyL = (unsigned int)max(ky - cutoff, 0.0f);
      unsigned int NyH = (unsigned int)min(ky + cutoff, (float)(size_y-1));

      unsigned int NzL = (unsigned int)max(kz - cutoff, 0.0f);
      unsigned int NzH = (unsigned int)min(kz + cutoff, (float)(size_z-1));

      if((pt.real != 0.0f || pt.imag != 0.0f) && pt.sdc != 0.0f)
      {
        float * __restrict dz2;
        float * __restrict dx2;
        float * __restrict dy2;

        int nx;
        int ny;
        int nz;

        for(dz2 = Dz2, nz=(int)NzL; (unsigned int)nz<=NzH; ++nz, ++dz2)
        {
          const float dz = kz-(float)nz;
          *dz2 = dz*dz;
        }
        for(dx2=Dx2,nx=(int)NxL; (unsigned int)nx<=NxH; ++nx,++dx2)
        {
          const float dx = kx-(float)nx;
          *dx2 = dx*dx;
        }
        for(dy2=Dy2, ny=(int)NyL; (unsigned int)ny<=NyH; ++ny,++dy2)
        {
          const float dy = ky-(float)ny;
          *dy2 = dy*dy;
        }

        unsigned int idxZ = (NzL-1U)*size_x*size_y;
        for(dz2=Dz2, nz=(int)NzL; (unsigned int)nz<=NzH; ++nz, ++dz2)
        {
          /* linear offset into 3-D matrix to get to zposition */
          idxZ += size_x*size_y;

          unsigned int idxY = (NyL-1U)*size_x;

          /* loop over x indexes, but only if curent distance is close enough (distance will increase by adding x&y distance) */
          if((*dz2)<cutoff2)
          {
            for(dy2=Dy2, ny=(int)NyL; (unsigned int)ny<=NyH; ++ny, ++dy2)
            {
              /* linear offset IN ADDITION to idxZ to get to Y position */
              idxY += size_x;

              const float dy2dz2=(*dz2)+(*dy2);

              unsigned int idx0 = idxY + idxZ;

              /* loop over y indexes, but only if curent distance is close enough (distance will increase by adding y distance) */
              if(dy2dz2<cutoff2)
              {
                for(dx2=Dx2, nx=(int)NxL; (unsigned int)nx<=NxH; ++nx, ++dx2)
                {
                  /* value to evaluate kernel at */
                  const float v = dy2dz2+(*dx2);

                  if(v<cutoff2)
                  {
                    /* linear index of (x,y,z) point */
                    const unsigned int idx = (unsigned int)nx + idx0;

                    /* kernel weighting value */
                    float w;
                    if (useLUT){
                      w = kernel_value_LUT(v, LUT, (int)sizeLUT, _1overCutoff2) * pt.sdc;
                    } else {
                      w = kernel_value_CPU(beta*sqrtf(1.0f-(v*_1overCutoff2))) * pt.sdc;
                    }

                    /* grid data */
                    #pragma omp atomic
                    gridData[idx].real += (w*pt.real);
                    #pragma omp atomic
                    gridData[idx].imag += (w*pt.imag);

                    /* estimate sample density */
                    #pragma omp atomic
                    sampleDensity[idx] += 1.0f;
                  }
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
