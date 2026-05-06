#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "UDTypes.h"

#define max(x,y) ((x<y)?y:x)
#define min(x,y) ((x>y)?y:x)

#define PI 3.14159265359f

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
  float cutoff2 = (width*width)*0.25f;

  unsigned int size;

  if(width > 0){
    // compute size of LUT based on kernel width
    size = (unsigned int)(10000*width);

    // allocate memory
    (*LUT) = (float*) malloc (size*sizeof(float));

    float inv_cutoff2 = 1.0f / cutoff2;
    float inv_size = 1.0f / (float)size;

    unsigned int k;
    for(k=0; k<size; ++k){
      // compute value to evaluate kernel at
      // v in the range 0:(_width/2)^2
      v = ((float)k * inv_size) * cutoff2;

      // compute kernel value and store
      (*LUT)[k] = kernel_value_CPU(beta*sqrtf(1.0f-(v*inv_cutoff2)));
    }
    (*sizeLUT) = size;
  }
}

static inline float kernel_value_LUT(float v, float* LUT, int sizeLUT, float _1overCutoff2)
{
  float fk = v * (float)sizeLUT * _1overCutoff2;
  unsigned int k0 = (unsigned int)fk;
  float alpha = fk - (float)k0;
  return LUT[k0] + alpha * (LUT[k0+1] - LUT[k0]);
}

int gridding_Gold(unsigned int n, parameters params, ReconstructionSample* sample, float* LUT, unsigned int sizeLUT, cmplx* gridData, float* sampleDensity){

  unsigned int size_x = params.gridSize[0];
  unsigned int size_y = params.gridSize[1];
  unsigned int size_z = params.gridSize[2];

  float cutoff = ((float)(params.kernelWidth))*0.5f; // cutoff radius
  float cutoff2 = cutoff*cutoff;                     // square of cutoff radius
  float _1overCutoff2 = 1.0f/cutoff2;                // 1 over square of cutoff radius

  float beta = PI * sqrtf(4.0f*params.kernelWidth*params.kernelWidth/(params.oversample*params.oversample) * (params.oversample-0.5f)*(params.oversample-0.5f)-0.8f);

  int useLUT = params.useLUT;

  #pragma omp parallel
  {
    float Dx2[100];
    float Dy2[100];
    float Dz2[100];

    #pragma omp for schedule(dynamic, 64)
    for (int i=0; i < (int)n; i++){
      ReconstructionSample pt = sample[i];

      float kx = pt.kX;
      float ky = pt.kY;
      float kz = pt.kZ;
      float real_val = pt.real;
      float imag_val = pt.imag;
      float sdc = pt.sdc;

      if((real_val == 0.0f && imag_val == 0.0f) || sdc == 0.0f)
        continue;

      unsigned int NxL = (unsigned int)max((kx - cutoff), 0.0f);
      unsigned int NxH = (unsigned int)min((kx + cutoff), (float)(size_x-1));

      unsigned int NyL = (unsigned int)max((ky - cutoff), 0.0f);
      unsigned int NyH = (unsigned int)min((ky + cutoff), (float)(size_y-1));

      unsigned int NzL = (unsigned int)max((kz - cutoff), 0.0f);
      unsigned int NzH = (unsigned int)min((kz + cutoff), (float)(size_z-1));

      unsigned int nz_count = NzH - NzL + 1;
      unsigned int nx_count = NxH - NxL + 1;
      unsigned int ny_count = NyH - NyL + 1;

      for(unsigned int dz_idx=0; dz_idx < nz_count; ++dz_idx)
      {
        float dz = (float)(NzL + dz_idx) - kz;
        Dz2[dz_idx] = dz * dz;
      }
      for(unsigned int dx_idx=0; dx_idx < nx_count; ++dx_idx)
      {
        float dx = (float)(NxL + dx_idx) - kx;
        Dx2[dx_idx] = dx * dx;
      }
      for(unsigned int dy_idx=0; dy_idx < ny_count; ++dy_idx)
      {
        float dy = (float)(NyL + dy_idx) - ky;
        Dy2[dy_idx] = dy * dy;
      }

      for(unsigned int dz_idx=0; dz_idx < nz_count; ++dz_idx)
      {
        float dz2_val = Dz2[dz_idx];
        if(dz2_val >= cutoff2) continue;

        unsigned int nz = NzL + dz_idx;
        unsigned int idxZ = nz * size_x * size_y;

        for(unsigned int dy_idx=0; dy_idx < ny_count; ++dy_idx)
        {
          float dy2dz2 = dz2_val + Dy2[dy_idx];
          if(dy2dz2 >= cutoff2) continue;

          unsigned int ny = NyL + dy_idx;
          unsigned int idxY = ny * size_x;
          unsigned int idx0 = idxY + idxZ;

          for(unsigned int dx_idx=0; dx_idx < nx_count; ++dx_idx)
          {
            float v = dy2dz2 + Dx2[dx_idx];
            if(v >= cutoff2) continue;

            unsigned int nx = NxL + dx_idx;
            unsigned int idx = nx + idx0;

            float w;
            if (useLUT){
              w = kernel_value_LUT(v, LUT, sizeLUT, _1overCutoff2) * sdc;
            } else {
              w = kernel_value_CPU(beta*sqrtf(1.0f-(v*_1overCutoff2))) * sdc;
            }

            #pragma omp atomic
            gridData[idx].real += (w*real_val);
            #pragma omp atomic
            gridData[idx].imag += (w*imag_val);
            #pragma omp atomic
            sampleDensity[idx] += 1.0f;
          }
        }
      }
    }
  }

  return 0;
}
