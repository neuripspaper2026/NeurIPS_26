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
  float cutoff2 = (width*width)/4.0f;

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
      (*LUT)[k] = kernel_value_CPU(beta*sqrtf(1.0f-(v/cutoff2)));
    }
    (*sizeLUT) = size;
  }
}

static inline float kernel_value_LUT(float v, float* LUT, int sizeLUT, float _1overCutoff2)
{
  float vScaled = v * (float)sizeLUT * _1overCutoff2;
  unsigned int k0 = (unsigned int)(vScaled);
  float frac = vScaled - (float)k0;
  return  LUT[k0] + frac * (LUT[k0+1] - LUT[k0]);
}

int gridding_Gold(unsigned int n, parameters params, ReconstructionSample* sample, float* LUT, unsigned int sizeLUT, cmplx* gridData, float* sampleDensity){

  unsigned int size_x = params.gridSize[0];
  unsigned int size_y = params.gridSize[1];
  unsigned int size_z = params.gridSize[2];

  float cutoff = ((float)(params.kernelWidth))/2.0f; // cutoff radius
  float cutoff2 = cutoff*cutoff;                    // square of cutoff radius
  float _1overCutoff2 = 1.0f/cutoff2;                  // 1 over square of cutoff radius

  float beta = PI * sqrtf(4.0f*params.kernelWidth*params.kernelWidth/(params.oversample*params.oversample) * (params.oversample-0.5f)*(params.oversample-0.5f)-0.8f);

  unsigned int gridSize = size_x * size_y * size_z;

  #pragma omp parallel
  {
    // Thread-local buffers
    float Dx2[100];
    float Dy2[100];
    float Dz2[100];
    
    // Thread-local accumulation buffers
    cmplx* local_gridData = (cmplx*) calloc(gridSize, sizeof(cmplx));
    float* local_sampleDensity = (float*) calloc(gridSize, sizeof(float));

    #pragma omp for schedule(dynamic, 64) nowait
    for (int i=0; i < (int)n; i++){
      ReconstructionSample pt = sample[i];

      float kx = pt.kX;
      float ky = pt.kY;
      float kz = pt.kZ;

      if((pt.real == 0.0f && pt.imag == 0.0f) || pt.sdc == 0.0f)
        continue;

      unsigned int NxL = max((kx - cutoff), 0.0f);
      unsigned int NxH = min((kx + cutoff), size_x-1.0f);

      unsigned int NyL = max((ky - cutoff), 0.0f);
      unsigned int NyH = min((ky + cutoff), size_y-1.0f);

      unsigned int NzL = max((kz - cutoff), 0.0f);
      unsigned int NzH = min((kz + cutoff), size_z-1.0f);

      // Precompute squared distances
      int nz_count = NzH - NzL + 1;
      int nx_count = NxH - NxL + 1;
      int ny_count = NyH - NyL + 1;

      for(int idx_z = 0; idx_z < nz_count; ++idx_z) {
        float dz = kz - (NzL + idx_z);
        Dz2[idx_z] = dz * dz;
      }
      for(int idx_x = 0; idx_x < nx_count; ++idx_x) {
        float dx = kx - (NxL + idx_x);
        Dx2[idx_x] = dx * dx;
      }
      for(int idx_y = 0; idx_y < ny_count; ++idx_y) {
        float dy = ky - (NyL + idx_y);
        Dy2[idx_y] = dy * dy;
      }

      float pt_real = pt.real;
      float pt_imag = pt.imag;
      float pt_sdc = pt.sdc;

      for(int idx_z = 0; idx_z < nz_count; ++idx_z) {
        float dz2 = Dz2[idx_z];
        if(dz2 >= cutoff2) continue;

        unsigned int nz = NzL + idx_z;
        unsigned int idxZ = nz * size_x * size_y;

        for(int idx_y = 0; idx_y < ny_count; ++idx_y) {
          float dy2dz2 = dz2 + Dy2[idx_y];
          if(dy2dz2 >= cutoff2) continue;

          unsigned int ny = NyL + idx_y;
          unsigned int idxY = ny * size_x;
          unsigned int idx0 = idxY + idxZ;

          for(int idx_x = 0; idx_x < nx_count; ++idx_x) {
            float v = dy2dz2 + Dx2[idx_x];
            if(v >= cutoff2) continue;

            unsigned int nx = NxL + idx_x;
            unsigned int idx = nx + idx0;

            float w;
            if (params.useLUT){
              w = kernel_value_LUT(v, LUT, sizeLUT, _1overCutoff2) * pt_sdc;
            } else {
              w = kernel_value_CPU(beta*sqrtf(1.0f-(v*_1overCutoff2))) * pt_sdc;
            }

            local_gridData[idx].real += (w * pt_real);
            local_gridData[idx].imag += (w * pt_imag);
            local_sampleDensity[idx] += 1.0f;
          }
        }
      }
    }

    // Reduction phase
    #pragma omp critical
    {
      for(unsigned int idx = 0; idx < gridSize; ++idx) {
        gridData[idx].real += local_gridData[idx].real;
        gridData[idx].imag += local_gridData[idx].imag;
        sampleDensity[idx] += local_sampleDensity[idx];
      }
    }

    free(local_gridData);
    free(local_sampleDensity);
  }

  return 0;
}
