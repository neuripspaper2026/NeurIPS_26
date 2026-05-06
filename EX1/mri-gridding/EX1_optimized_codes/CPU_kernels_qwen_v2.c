<<<CODE>>>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "UDTypes.h"

#define max(x,y) ((x<y)?y:x)
#define min(x,y) ((x>y)?y:x)

#define PI 3.14159265359f

static inline float kernel_value_CPU(float v){
  const float z = v*v;

  // polynomials taken from http://ccrma.stanford.edu/CCRMA/Courses/422/projects/kbd/kbdwindow.cpp
  const float numerator_coeffs[] = {
    0.210580722890567e-22f, 0.380715242345326e-19f, 0.479440257548300e-16f,
    0.435125971262668e-13f, 0.300931127112960e-10f, 0.160224679395361e-7f,
    0.654858370096785e-5f,  0.202591084143397e-2f,  0.463076284721000e0f,
    0.754337328948189e2f,   0.830792541809429e4f,   0.571661130563785e6f,
    0.216415572361227e8f,   0.356644482244025e9f,   0.144048298227235e10f
  };

  float num = numerator_coeffs[0];
  for (int i = 1; i < 15; i++) {
    num = num * z + numerator_coeffs[i];
  }

  const float denominator_coeffs[] = { 
    -0.307646912682801e4f, 0.347626332405882e7f, -0.144048298227235e10f
  };
  
  float den = z * (z * (z + denominator_coeffs[0]) + denominator_coeffs[1]) + denominator_coeffs[2];

  return -num/den;
}

void calculateLUT(float beta, float width, float** LUT, unsigned int* sizeLUT){
  const float cutoff2 = (width*width)/4.0f;

  if(width > 0){
    // compute size of LUT based on kernel width
    const unsigned int size = (unsigned int)(10000*width);

    // allocate memory
    (*LUT) = (float*) malloc (size*sizeof(float));

    const float scale = cutoff2 / (float)size;
    
    unsigned int k;
    for(k=0; k<size; ++k){
      // compute value to evaluate kernel at
      // v in the range 0:(_width/2)^2
      const float v = ((float)k) * scale;

      // compute kernel value and store
      (*LUT)[k] = kernel_value_CPU(beta*sqrtf(1.0f-(v/cutoff2)));
    }
    (*sizeLUT) = size;
  }
}

static inline float kernel_value_LUT(float v, float* LUT, int sizeLUT, float _1overCutoff2)
{
  const float scaled_v = v * (float)sizeLUT * _1overCutoff2;
  const unsigned int k0 = (unsigned int)scaled_v;
  const float v0 = ((float)k0) / _1overCutoff2;
  return  LUT[k0] + ((scaled_v - (float)k0) * (LUT[k0+1]-LUT[k0]));
}

int gridding_Gold(unsigned int n, parameters params, ReconstructionSample* sample, float* LUT, unsigned int sizeLUT, cmplx* gridData, float* sampleDensity){

  const unsigned int size_x = params.gridSize[0];
  const unsigned int size_y = params.gridSize[1];
  const unsigned int size_z = params.gridSize[2];

  const float cutoff = ((float)(params.kernelWidth))/2.0f; // cutoff radius
  const float cutoff2 = cutoff*cutoff;                    // square of cutoff radius
  const float _1overCutoff2 = 1.0f/cutoff2;               // 1 over square of cutoff radius

  const float beta = PI * sqrtf(4*params.kernelWidth*params.kernelWidth/(params.oversample*params.oversample) * (params.oversample-0.5f)*(params.oversample-0.5f)-0.8f);

  int i;
  for (i=0; i < n; i++){
    const ReconstructionSample pt = sample[i];

    if(pt.real == 0.0f && pt.imag == 0.0f) continue;
    if(pt.sdc == 0.0f) continue;

    const float kx = pt.kX;
    const float ky = pt.kY;
    const float kz = pt.kZ;

    const unsigned int NxL = max((kx - cutoff), 0.0f);
    const unsigned int NxH = min((kx + cutoff), (float)(size_x-1));

    const unsigned int NyL = max((ky - cutoff), 0.0f);
    const unsigned int NyH = min((ky + cutoff), (float)(size_y-1));

    const unsigned int NzL = max((kz - cutoff), 0.0f);
    const unsigned int NzH = min((kz + cutoff), (float)(size_z-1));

    float Dz2[100];
    float Dy2[100];
    float Dx2[100];

    unsigned int nz;
    for(nz=NzL; nz<=NzH; ++nz)
    {
      Dz2[nz-NzL] = ((kz-nz)*(kz-nz));
    }
    
    unsigned int nx;
    for(nx=NxL; nx<=NxH; ++nx)
    {
      Dx2[nx-NxL] = ((kx-nx)*(kx-nx));
    }
    
    unsigned int ny;
    for(ny=NyL; ny<=NyH; ++ny)
    {
      Dy2[ny-NyL] = ((ky-ny)*(ky-ny));
    }

    const unsigned int idxZ_base = (NzL-1)*size_x*size_y;
    unsigned int dz_idx = 0;
    for(nz=NzL; nz<=NzH; ++nz, ++dz_idx)
    {
      /* linear offset into 3-D matrix to get to zposition */
      const unsigned int idxZ = idxZ_base + (nz+1)*size_x*size_y;

      const float dz2_val = Dz2[dz_idx];
      
      /* loop over x indexes, but only if current distance is close enough (distance will increase by adding x&y distance) */
      if(dz2_val < cutoff2)
      {
        const unsigned int idxY_base = (NyL-1)*size_x;
        unsigned int dy_idx = 0;
        for(ny=NyL; ny<=NyH; ++ny, ++dy_idx)
        {
          /* linear offset IN ADDITION to idxZ to get to Y position */
          const unsigned int idxY = idxY_base + (ny+1)*size_x;

          const float dy2_val = Dy2[dy_idx];
          const float dy2dz2 = dz2_val + dy2_val;

          const unsigned int idx0 = idxY + idxZ;

          /* loop over y indexes, but only if current distance is close enough (distance will increase by adding y distance) */
          if(dy2dz2 < cutoff2)
          {
            unsigned int dx_idx = 0;
            for(nx=NxL; nx<=NxH; ++nx, ++dx_idx)
            {
              /* value to evaluate kernel at */
              const float dx2_val = Dx2[dx_idx];
              const float v = dy2dz2 + dx2_val;

              if(v < cutoff2)
              {
                /* linear index of (x,y,z) point */
                const unsigned int idx = nx + idx0;

                /* kernel weighting value */
                float w;
                if (params.useLUT){
                  w = kernel_value_LUT(v, LUT, sizeLUT, _1overCutoff2) * pt.sdc;
                } else {
                  w = kernel_value_CPU(beta*sqrtf(1.0f-(v*_1overCutoff2))) * pt.sdc;
                }

                /* grid data */
                gridData[idx].real += (w
