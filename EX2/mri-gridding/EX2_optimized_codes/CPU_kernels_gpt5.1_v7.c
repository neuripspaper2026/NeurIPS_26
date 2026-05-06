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

  /* Polynomials taken from http://ccrma.stanford.edu/CCRMA/Courses/422/projects/kbd/kbdwindow.cpp */
  const float num =
    (float)(
      z*( z*( z*( z*( z*( z*( z*( z*( z*( z*( z*( z*( z*
      0.210580722890567e-22f  + 0.380715242345326e-19f ) +
       0.479440257548300e-16f) + 0.435125971262668e-13f ) +
       0.300931127112960e-10f) + 0.160224679395361e-7f  ) +
       0.654858370096785e-5f)  + 0.202591084143397e-2f  ) +
       0.463076284721000e0f)   + 0.754337328948189e2f   ) +
       0.830792541809429e4f)   + 0.571661130563785e6f   ) +
       0.216415572361227e8f)   + 0.356644482244025e9f   ) +
       0.144048298227235e10f);

  const float den =
    (float)(z*(z*(z-0.307646912682801e4f)+0.347626332405882e7f)-0.144048298227235e10f);

  return -num/den;
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

    const float inv_size = 1.0f/(float)size;

    unsigned int k;
    for(k=0; k<size; ++k){
      /* compute value to evaluate kernel at
       * v in the range 0:(_width/2)^2
       */
      v = ((float)k)*inv_size*cutoff2;

      /* compute kernel value and store */
      (*LUT)[k] = kernel_value_CPU(beta*sqrtf(1.0f-(v/cutoff2)));
    }
    (*sizeLUT) = size;
  }
}

static inline float kernel_value_LUT(float v, float* LUT, int sizeLUT, float _1overCutoff2)
{
  /* v is squared distance in [0, cutoff2] */
  const float scale = (float)sizeLUT*_1overCutoff2;
  const float vk = v*scale;
  int k0 = (int)vk;
  if (k0 < 0) k0 = 0;
  if (k0 >= sizeLUT-1) k0 = sizeLUT-2;

  const float v0 = (float)k0/_1overCutoff2;
  return  LUT[k0] + ((v-v0)*(LUT[k0+1]-LUT[k0])*_1overCutoff2);
}

int gridding_Gold(unsigned int n, parameters params, ReconstructionSample* sample, float* LUT, unsigned int sizeLUT, cmplx* gridData, float* sampleDensity){

  const unsigned int size_x = (unsigned int)params.gridSize[0];
  const unsigned int size_y = (unsigned int)params.gridSize[1];
  const unsigned int size_z = (unsigned int)params.gridSize[2];

  const float cutoff  = ((float)(params.kernelWidth))*0.5f; /* cutoff radius           */
  const float cutoff2 = cutoff*cutoff;                      /* square of cutoff radius */
  const float _1overCutoff2 = 1.0f/cutoff2;                 /* 1 over square of cutoff radius */

  const float oversample = params.oversample;
  const float kw = (float)params.kernelWidth;
  const float beta = PI * sqrtf(4.0f*kw*kw/(oversample*oversample) *
                                (oversample-0.5f)*(oversample-0.5f)-0.8f);

  /* Parallelize over samples.
   * Updates to gridData and sampleDensity are atomic to keep correctness
   * without duplicating arrays.
   */
  #pragma omp parallel
  {
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
    float *dx2 = NULL;
    float *dy2 = NULL;
    float *dz2 = NULL;

    float dy2dz2;
    float v;

    #pragma omp for schedule(static)
    for (int i = 0; i < (int)n; i++){
      const ReconstructionSample pt = sample[i];

      const float kx = pt.kX;
      const float ky = pt.kY;
      const float kz = pt.kZ;

      if( (pt.real == 0.0f && pt.imag == 0.0f) || pt.sdc == 0.0f )
        continue;

      const float kx_minus_cutoff = kx - cutoff;
      const float kx_plus_cutoff  = kx + cutoff;
      const float ky_minus_cutoff = ky - cutoff;
      const float ky_plus_cutoff  = ky + cutoff;
      const float kz_minus_cutoff = kz - cutoff;
      const float kz_plus_cutoff  = kz + cutoff;

      NxL = (unsigned int)max(kx_minus_cutoff, 0.0f);
      NxH = (unsigned int)min(kx_plus_cutoff,  (float)(size_x-1U));

      NyL = (unsigned int)max(ky_minus_cutoff, 0.0f);
      NyH = (unsigned int)min(ky_plus_cutoff,  (float)(size_y-1U));

      NzL = (unsigned int)max(kz_minus_cutoff, 0.0f);
      NzH = (unsigned int)min(kz_plus_cutoff,  (float)(size_z-1U));

      const unsigned int rangeX = NxH - NxL + 1U;
      const unsigned int rangeY = NyH - NyL + 1U;
      const unsigned int rangeZ = NzH - NzL + 1U;

      /* ensure temp arrays are large enough (kernelWidth is small in practice) */
      if (rangeX > 100U || rangeY > 100U || rangeZ > 100U)
        continue;

      dz2 = Dz2;
      for(nz=(int)NzL; nz<=(int)NzH; ++nz, ++dz2)
      {
        const float dz = kz-(float)nz;
        *dz2 = dz*dz;
      }
      dx2 = Dx2;
      for(nx=(int)NxL; nx<=(int)NxH; ++nx,++dx2)
      {
        const float dx = kx-(float)nx;
        *dx2 = dx*dx;
      }
      dy2 = Dy2;
      for(ny=(int)NyL; ny<=(int)NyH; ++ny,++dy2)
      {
        const float dy = ky-(float)ny;
        *dy2 = dy*dy;
      }

      idxZ = (NzL-1U)*size_x*size_y;
      dz2 = Dz2;
      for(nz=(int)NzL; nz<=(int)NzH; ++nz, ++dz2)
      {
        /* linear offset into 3-D matrix to get to zposition */
        idxZ += size_x*size_y;

        idxY = (NyL-1U)*size_x;

        /* loop over x indexes, but only if current distance is close enough
         * (distance will increase by adding x & y distance)
         */
        if((*dz2) < cutoff2)
        {
          dy2 = Dy2;
          for(ny=(int)NyL; ny<=(int)NyH; ++ny, ++dy2)
          {
            /* linear offset in addition to idxZ to get to Y position */
            idxY += size_x;

            dy2dz2 = (*dz2)+(*dy2);

            idx0 = idxY + idxZ;

            /* loop over x indexes, but only if current distance is close enough
             * (distance will increase by adding y distance)
             */
            if(dy2dz2 < cutoff2)
            {
              dx2 = Dx2;
              for(nx=(int)NxL; nx<=(int)NxH; ++nx, ++dx2)
              {
                /* value to evaluate kernel at */
                v = dy2dz2+(*dx2);

                if(v < cutoff2)
                {
                  /* linear index of (x,y,z) point */
                  idx = (unsigned int)nx + idx0;

                  /* kernel weighting value */
                  if (params.useLUT){
                    w = kernel_value_LUT(v, LUT, (int)sizeLUT, _1overCutoff2) * pt.sdc;
                  } else {
                    w = kernel_value_CPU(beta*sqrtf(1.0f-(v*_1overCutoff2))) * pt.sdc;
                  }

                  /* grid data: atomic updates to avoid races */
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

  return 0;
}
