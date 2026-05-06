#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

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
    float inv_size = 1.0f / ((float)size);
    
    unsigned int k;
    for(k=0; k<size; ++k){
      // compute value to evaluate kernel at
      // v in the range 0:(_width/2)^2
      v = (((float)k)*inv_size)*cutoff2;

      // compute kernel value and store
      (*LUT)[k] = kernel_value_CPU(beta*sqrtf(1.0f-(v*inv_cutoff2)));
    }
    (*sizeLUT) = size;
  }
}

static inline float kernel_value_LUT(float v, float* LUT, int sizeLUT, float _1overCutoff2)
{
  unsigned int k0;
  float v_scaled;

  v_scaled = v * (float)sizeLUT * _1overCutoff2;
  k0 = (unsigned int)(v_scaled);
  float v0 = (float)k0;
  float lut_k0 = LUT[k0];
  return  lut_k0 + ((v_scaled - v0) * (LUT[k0+1] - lut_k0));
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

  unsigned int size_x = params.gridSize[0];
  unsigned int size_y = params.gridSize[1];
  unsigned int size_z = params.gridSize[2];

  float cutoff = ((float)(params.kernelWidth))*0.5f; // cutoff radius
  float cutoff2 = cutoff*cutoff;                    // square of cutoff radius
  float _1overCutoff2 = 1.0f/cutoff2;                  // 1 over square of cutoff radius

  float beta = PI * sqrtf(4.0f*params.kernelWidth*params.kernelWidth/(params.oversample*params.oversample) * (params.oversample-0.5f)*(params.oversample-0.5f)-0.8f);

  unsigned int size_xy = size_x * size_y;
  float size_x_minus_1 = (float)(size_x - 1);
  float size_y_minus_1 = (float)(size_y - 1);
  float size_z_minus_1 = (float)(size_z - 1);

  int useLUT = params.useLUT;

  int i;
  for (i=0; i < n; i++){
    ReconstructionSample pt = sample[i];

    float kx = pt.kX;
    float ky = pt.kY;
    float kz = pt.kZ;
    float pt_real = pt.real;
    float pt_imag = pt.imag;
    float pt_sdc = pt.sdc;

    if((pt_real == 0.0f && pt_imag == 0.0f) || pt_sdc == 0.0f)
      continue;

    NxL = max((kx - cutoff), 0.0f);
    NxH = min((kx + cutoff), size_x_minus_1);

    NyL = max((ky - cutoff), 0.0f);
    NyH = min((ky + cutoff), size_y_minus_1);

    NzL = max((kz - cutoff), 0.0f);
    NzH = min((kz + cutoff), size_z_minus_1);

    unsigned int nz_count = NzH - NzL + 1;
    unsigned int nx_count = NxH - NxL + 1;
    unsigned int ny_count = NyH - NyL + 1;

    for(dz2 = Dz2, nz=NzL; nz<=NzH; ++nz, ++dz2)
    {
      float diff = kz - nz;
      *dz2 = diff * diff;
    }
    for(dx2=Dx2,nx=NxL; nx<=NxH; ++nx,++dx2)
    {
      float diff = kx - nx;
      *dx2 = diff * diff;
    }
    for(dy2=Dy2, ny=NyL; ny<=NyH; ++ny,++dy2)
    {
      float diff = ky - ny;
      *dy2 = diff * diff;
    }

    idxZ = NzL * size_xy;
    for(dz2=Dz2, nz=NzL; nz<=NzH; ++nz, ++dz2)
    {
      float dz2_val = *dz2;
      
      if(dz2_val >= cutoff2) {
        idxZ += size_xy;
        continue;
      }

      idxY = NyL * size_x;

      for(dy2=Dy2, ny=NyL; ny<=NyH; ++ny, ++dy2)
      {
        dy2dz2 = dz2_val + (*dy2);

        if(dy2dz2 >= cutoff2) {
          idxY += size_x;
          continue;
        }

        idx0 = idxY + idxZ;

        for(dx2=Dx2, nx=NxL; nx<=NxH; ++nx, ++dx2)
        {
          v = dy2dz2 + (*dx2);

          if(v < cutoff2)
          {
            idx = nx + idx0;

            if (useLUT){
              w = kernel_value_LUT(v, LUT, sizeLUT, _1overCutoff2) * pt_sdc;
            } else {
              w = kernel_value_CPU(beta*sqrtf(1.0f-(v*_1overCutoff2))) * pt_sdc;
            }

            gridData[idx].real += (w*pt_real);
            gridData[idx].imag += (w*pt_imag);

            sampleDensity[idx] += 1.0f;
          }
        }
        idxY += size_x;
      }
      idxZ += size_xy;
    }
  }
  return 0;
}
