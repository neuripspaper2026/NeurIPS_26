#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "UDTypes.h"

#define max(x,y) ((x<y)?y:x)
#define min(x,y) ((x>y)?y:x)

#define PI 3.14159265359

float kernel_value_CPU(float v){

  float rValue = 0;

  const float z = v*v;

  // polynomials taken from http://ccrma.stanford.edu/CCRMA/Courses/422/projects/kbd/kbdwindow.cpp
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

  rValue = -num/den;

  return rValue;
}

void calculateLUT(float beta, float width, float** LUT, unsigned int* sizeLUT){
  float v;
  float cutoff2 = (width*width)/4.0;

  unsigned int size;

  if(width > 0){
    // compute size of LUT based on kernel width
    size = (unsigned int)(10000*width);

    // allocate memory
    (*LUT) = (float*) malloc (size*sizeof(float));

    unsigned int k;
    for(k=0; k<size; ++k){
      // compute value to evaluate kernel at
      // v in the range 0:(_width/2)^2
      v = (((float)k)/((float)size))*cutoff2;

      // compute kernel value and store
      (*LUT)[k] = kernel_value_CPU(beta*sqrt(1.0-(v/cutoff2)));
    }
    (*sizeLUT) = size;
  }
}

float kernel_value_LUT(float v, float* LUT, int sizeLUT, float _1overCutoff2)
{
  unsigned int k0;
  float v0;

  v *= (float)sizeLUT;
  k0=(unsigned int)(v*_1overCutoff2);
  v0 = ((float)k0)/_1overCutoff2;
  return  LUT[k0] + ((v-v0)*(LUT[k0+1]-LUT[k0])/_1overCutoff2);
}

int gridding_Gold(unsigned int n, parameters params, ReconstructionSample* sample, float* LUT, unsigned int sizeLUT, cmplx* gridData, float* sampleDensity){

  unsigned int size_x = params.gridSize[0];
  unsigned int size_y = params.gridSize[1];
  unsigned int size_z = params.gridSize[2];

  float cutoff = ((float)(params.kernelWidth))/2.0; // cutoff radius
  float cutoff2 = cutoff*cutoff;                    // square of cutoff radius
  float _1overCutoff2 = 1/cutoff2;                  // 1 over square of cutoff radius

  float beta = PI * sqrt(4*params.kernelWidth*params.kernelWidth/(params.oversample*params.oversample) * (params.oversample-.5)*(params.oversample-.5)-.8);

  #pragma omp parallel for schedule(dynamic)
  for (int i=0; i < n; i++){
    ReconstructionSample pt = sample[i];

    float kx = pt.kX;
    float ky = pt.kY;
    float kz = pt.kZ;

    unsigned int NxL = max((kx - cutoff), 0.0);
    unsigned int NxH = min((kx + cutoff), size_x-1.0);

    unsigned int NyL = max((ky - cutoff), 0.0);
    unsigned int NyH = min((ky + cutoff), size_y-1.0);

    unsigned int NzL = max((kz - cutoff), 0.0);
    unsigned int NzH = min((kz + cutoff), size_z-1.0);

    if((pt.real != 0.0 || pt.imag != 0.0) && pt.sdc!=0.0)
    {
      // Declare thread-local arrays to avoid false sharing
      float Dx2[100];
      float Dy2[100];
      float Dz2[100];

      unsigned int idxZ = (NzL-1)*size_x*size_y;
      for(unsigned int nz=NzL; nz<=NzH; ++nz)
      {
        float dz2 = ((kz-nz)*(kz-nz));
        /* linear offset into 3-D matrix to get to zposition */
        idxZ += size_x*size_y;

        unsigned int idxY = (NyL-1)*size_x;

        /* loop over x indexes, but only if curent distance is close enough (distance will increase by adding x&y distance) */
        if(dz2<cutoff2)
        {
          for(unsigned int ny=NyL; ny<=NyH; ++ny)
          {
            float dy2 = ((ky-ny)*(ky-ny));
            /* linear offset IN ADDITION to idxZ to get to Y position */
            idxY += size_x;

            float dy2dz2= dz2 + dy2;

            unsigned int idx0 = idxY + idxZ;

            /* loop over y indexes, but only if curent distance is close enough (distance will increase by adding y distance) */
            if(dy2dz2<cutoff2)
            {
              for(unsigned int nx=NxL; nx<=NxH; ++nx)
              {
                float dx2 = ((kx-nx)*(kx-nx));
                /* value to evaluate kernel at */
                float v = dy2dz2 + dx2;

                if(v<cutoff2)
                {
                  /* linear index of (x,y,z) point */
                  unsigned int idx = nx + idx0;

                  /* kernel weighting value */
                  float w;
                  if (params.useLUT){
        		    w = kernel_value_LUT(v, LUT, sizeLUT, _1overCutoff2) * pt.sdc;
		          } else {
		            w = kernel_value_CPU(beta*sqrt(1.0-(v*_1overCutoff2))) * pt.sdc;
		          }

                  /* grid data */
                  #pragma omp atomic
                  gridData[idx].real += (w*pt.real);
                  #pragma omp atomic
                  gridData[idx].imag += (w*pt.imag);

                  /* estimate sample density */
                  #pragma omp atomic
                  sampleDensity[idx] += 1.0;
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
