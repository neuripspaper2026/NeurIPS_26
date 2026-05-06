#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "UDTypes.h"

#define max(x,y) ((x)<(y)?(y):(x))
#define min(x,y) ((x)>(y)?(y):(x))

#define PI 3.14159265359

static inline float kernel_value_CPU(float v){

  const float z = v*v;

  /* Horner form polynomial evaluation (unchanged mathematically) */
  float num =
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
  float v;
  const float cutoff2 = (width*width)*0.25f;

  unsigned int size;

  if(width > 0.0f){
    /* compute size of LUT based on kernel width */
    size = (unsigned int)(10000.0f*width);

    /* allocate memory */
    (*LUT) = (float*) malloc ((size_t)size*sizeof(float));

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

static inline float kernel_value_LUT(float v, float* LUT, int sizeLUT, float _1overCutoff2)
{
  /* v is in [0, cutoff2]; map to [0,sizeLUT] */
  const float scaled = v*(float)sizeLUT;
  const unsigned int k0 = (unsigned int)(scaled*_1overCutoff2);
  const float v0 = ((float)k0)/_1overCutoff2;
  return  LUT[k0] + ((scaled-v0)*(LUT[k0+1]-LUT[k0])/_1overCutoff2);
}

int gridding_Gold(unsigned int n, parameters params, ReconstructionSample* sample,
                  float* LUT, unsigned int sizeLUT, cmplx* gridData, float* sampleDensity){

  const unsigned int size_x = (unsigned int)params.gridSize[0];
  const unsigned int size_y = (unsigned int)params.gridSize[1];
  const unsigned int size_z = (unsigned int)params.gridSize[2];

  const float cutoff  = (float)(params.kernelWidth)*0.5f;   /* cutoff radius */
  const float cutoff2 = cutoff*cutoff;                      /* square of cutoff radius */
  const float _1overCutoff2 = 1.0f/cutoff2;                 /* 1 / cutoff^2 */

  const float kW = (float)params.kernelWidth;
  const float os = params.oversample;
  const float beta = PI * sqrtf(4.0f*kW*kW/(os*os) * (os-0.5f)*(os-0.5f)-0.8f);

  /* maximum half-width of support in each dimension */
  const unsigned int maxSupport =
    (unsigned int)(2.0f*cutoff + 2.0f); /* +2 safety margin */

  /* stack arrays sized to maximum support; large enough for original [0..99] */
  float *Dx2 = (float*)alloca(maxSupport*sizeof(float));
  float *Dy2 = (float*)alloca(maxSupport*sizeof(float));
  float *Dz2 = (float*)alloca(maxSupport*sizeof(float));

  /* parallel over samples; updates same grid cells, so use atomic to avoid races */
  #pragma omp parallel for default(none) shared(sample,gridData,sampleDensity,LUT,params) \
                           shared(size_x,size_y,size_z,n,cutoff,cutoff2,_1overCutoff2,beta,maxSupport,sizeLUT)
  for (int i = 0; (unsigned int)i < n; i++){
    ReconstructionSample pt = sample[i];

    const float kx = pt.kX;
    const float ky = pt.kY;
    const float kz = pt.kZ;

    unsigned int NxL = (unsigned int)max(kx - cutoff, 0.0f);
    unsigned int NxH = (unsigned int)min(kx + cutoff, (float)(size_x-1u));

    unsigned int NyL = (unsigned int)max(ky - cutoff, 0.0f);
    unsigned int NyH = (unsigned int)min(ky + cutoff, (float)(size_y-1u));

    unsigned int NzL = (unsigned int)max(kz - cutoff, 0.0f);
    unsigned int NzH = (unsigned int)min(kz + cutoff, (float)(size_z-1u));

    if ((pt.real != 0.0f || pt.imag != 0.0f) && pt.sdc != 0.0f)
    {
      float *dx2;
      float *dy2;
      float *dz2;

      int nx, ny, nz;

      /* precompute squared distances */
      for(dz2 = Dz2, nz=(int)NzL; (unsigned int)nz<=NzH; ++nz, ++dz2)
      {
        const float dz = kz-(float)nz;
        *dz2 = dz*dz;
      }
      for(dx2=Dx2, nx=(int)NxL; (unsigned int)nx<=NxH; ++nx,++dx2)
      {
        const float dx = kx-(float)nx;
        *dx2 = dx*dx;
      }
      for(dy2=Dy2, ny=(int)NyL; (unsigned int)ny<=NyH; ++ny,++dy2)
      {
        const float dy = ky-(float)ny;
        *dy2 = dy*dy;
      }

      unsigned int idxZ = (NzL-1u)*size_x*size_y;
      for(dz2=Dz2, nz=(int)NzL; (unsigned int)nz<=NzH; ++nz, ++dz2)
      {
        /* linear offset into 3-D matrix to get to zposition */
        idxZ += size_x*size_y;

        unsigned int idxY = (NyL-1u)*size_x;

        if((*dz2)<cutoff2)
        {
          for(dy2=Dy2, ny=(int)NyL; (unsigned int)ny<=NyH; ++ny, ++dy2)
          {
            /* linear offset IN ADDITION to idxZ to get to Y position */
            idxY += size_x;

            const float dy2dz2 = (*dz2)+(*dy2);

            const unsigned int idx0 = idxY + idxZ;

            if(dy2dz2<cutoff2)
            {
              for(dx2=Dx2, nx=(int)NxL; (unsigned int)nx<=NxH; ++nx, ++dx2)
              {
                const float v = dy2dz2+(*dx2);

                if(v<cutoff2)
                {
                  const unsigned int idx = (unsigned int)nx + idx0;

                  float w;
                  if (params.useLUT){
                    w = kernel_value_LUT(v, LUT, (int)sizeLUT, _1overCutoff2) * pt.sdc;
                  } else {
                    w = kernel_value_CPU(beta*sqrtf(1.0f-(v*_1overCutoff2))) * pt.sdc;
                  }

                  /* atomic updates to avoid data races between threads */
                  #pragma omp atomic
                  gridData[idx].real += (w*pt.real);
                  #pragma omp atomic
                  gridData[idx].imag += (w*pt.imag);
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
