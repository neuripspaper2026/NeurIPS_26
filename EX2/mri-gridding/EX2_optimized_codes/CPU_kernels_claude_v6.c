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
      v = ((float)k) * inv_size * cutoff2;

      // compute kernel value and store
      (*LUT)[k] = kernel_value_CPU(beta*sqrtf(1.0f-(v*inv_cutoff2)));
    }
    (*sizeLUT) = size;
  }
}

static inline float kernel_value_LUT(float v, float* LUT, int sizeLUT, float _1overCutoff2)
{
  float scaled_v = v * (float)sizeLUT * _1overCutoff2;
  unsigned int k0 = (unsigned int)(scaled_v);
  float frac = scaled_v - (float)k0;
  return  LUT[k0] + frac * (LUT[k0+1] - LUT[k0]);
}

int gridding_Gold(unsigned int n, parameters params, ReconstructionSample* sample, float* LUT, unsigned int sizeLUT, cmplx* gridData, float* sampleDensity){

  unsigned int size_x = params.gridSize[0];
  unsigned int size_y = params.gridSize[1];
  unsigned int size_z = params.gridSize[2];

  float cutoff = ((float)(params.kernelWidth))*0.5f; // cutoff radius
  float cutoff2 = cutoff*cutoff;                    // square of cutoff radius
  float _1overCutoff2 = 1.0f/cutoff2;                  // 1 over square of cutoff radius

  float beta = PI * sqrtf(4.0f*params.kernelWidth*params.kernelWidth/(params.oversample*params.oversample) * (params.oversample-0.5f)*(params.oversample-0.5f)-0.8f);

  unsigned int grid_size = size_x * size_y * size_z;

  #pragma omp parallel
  {
    // Thread-local buffers
    float Dx2_local[100];
    float Dy2_local[100];
    float Dz2_local[100];
    
    // Thread-local accumulators for gridData and sampleDensity
    cmplx* gridData_local = (cmplx*) calloc(grid_size, sizeof(cmplx));
    float* sampleDensity_local = (float*) calloc(grid_size, sizeof(float));

    #pragma omp for schedule(dynamic, 64) nowait
    for (int i = 0; i < (int)n; i++){
      ReconstructionSample pt = sample[i];

      float kx = pt.kX;
      float ky = pt.kY;
      float kz = pt.kZ;

      if((pt.real == 0.0f && pt.imag == 0.0f) || pt.sdc == 0.0f)
        continue;

      unsigned int NxL = (unsigned int)max((kx - cutoff), 0.0f);
      unsigned int NxH = (unsigned int)min((kx + cutoff), size_x-1.0f);

      unsigned int NyL = (unsigned int)max((ky - cutoff), 0.0f);
      unsigned int NyH = (unsigned int)min((ky + cutoff), size_y-1.0f);

      unsigned int NzL = (unsigned int)max((kz - cutoff), 0.0f);
      unsigned int NzH = (unsigned int)min((kz + cutoff), size_z-1.0f);

      // Precompute squared distances
      unsigned int nx_count = NxH - NxL + 1;
      unsigned int ny_count = NyH - NyL + 1;
      unsigned int nz_count = NzH - NzL + 1;

      for(unsigned int idx = 0; idx < nz_count; ++idx) {
        float diff = kz - (NzL + idx);
        Dz2_local[idx] = diff * diff;
      }
      for(unsigned int idx = 0; idx < nx_count; ++idx) {
        float diff = kx - (NxL + idx);
        Dx2_local[idx] = diff * diff;
      }
      for(unsigned int idx = 0; idx < ny_count; ++idx) {
        float diff = ky - (NyL + idx);
        Dy2_local[idx] = diff * diff;
      }

      unsigned int idxZ_base = NzL * size_x * size_y;
      
      for(unsigned int dz_idx = 0; dz_idx < nz_count; ++dz_idx) {
        float dz2 = Dz2_local[dz_idx];
        
        if(dz2 >= cutoff2)
          continue;

        unsigned int idxZ = idxZ_base + dz_idx * size_x * size_y;
        unsigned int idxY_base = NyL * size_x;

        for(unsigned int dy_idx = 0; dy_idx < ny_count; ++dy_idx) {
          float dy2dz2 = dz2 + Dy2_local[dy_idx];

          if(dy2dz2 >= cutoff2)
            continue;

          unsigned int idxY = idxY_base + dy_idx * size_x;
          unsigned int idx0 = idxY + idxZ;

          for(unsigned int dx_idx = 0; dx_idx < nx_count; ++dx_idx) {
            float v = dy2dz2 + Dx2_local[dx_idx];

            if(v < cutoff2) {
              unsigned int idx = NxL + dx_idx + idx0;

              float w;
              if (params.useLUT){
                w = kernel_value_LUT(v, LUT, sizeLUT, _1overCutoff2) * pt.sdc;
              } else {
                w = kernel_value_CPU(beta*sqrtf(1.0f-(v*_1overCutoff2))) * pt.sdc;
              }

              gridData_local[idx].real += (w*pt.real);
              gridData_local[idx].imag += (w*pt.imag);
              sampleDensity_local[idx] += 1.0f;
            }
          }
        }
      }
    }

    // Reduction phase
    #pragma omp critical
    {
      for(unsigned int idx = 0; idx < grid_size; ++idx) {
        gridData[idx].real += gridData_local[idx].real;
        gridData[idx].imag += gridData_local[idx].imag;
        sampleDensity[idx] += sampleDensity_local[idx];
      }
    }

    free(gridData_local);
    free(sampleDensity_local);
  }

  return 0;
}
