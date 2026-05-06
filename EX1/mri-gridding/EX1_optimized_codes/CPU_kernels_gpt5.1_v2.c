#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "UDTypes.h"

#define max(x,y) ((x<y)?y:x)
#define min(x,y) ((x>y)?y:x)

#define PI 3.14159265359

static inline float kernel_value_CPU(float v){
  const float z = v * v;

  const float num =
    z * (z * (z * (z * (z * (z * (z * (z * (z * (z * (z * (z * (z *
      (z * 0.210580722890567e-22f  + 0.380715242345326e-19f ) +
       0.479440257548300e-16f) + 0.435125971262668e-13f ) +
       0.300931127112960e-10f) + 0.160224679395361e-7f  ) +
       0.654858370096785e-5f)  + 0.202591084143397e-2f  ) +
       0.463076284721000e0f)   + 0.754337328948189e2f   ) +
       0.830792541809429e4f)   + 0.571661130563785e6f   ) +
       0.216415572361227e8f)   + 0.356644482244025e9f   ) +
       0.144048298227235e10f);

  const float den = z * (z * (z - 0.307646912682801e4f) +
                         0.347626332405882e7f) -
                    0.144048298227235e10f;

  return -num / den;
}

void calculateLUT(float beta, float width, float** LUT, unsigned int* sizeLUT){
  const float cutoff2 = (width * width) * 0.25f;

  unsigned int size;

  if (width > 0.0f){
    /* compute size of LUT based on kernel width */
    size = (unsigned int)(10000.0f * width);

    /* allocate memory */
    *LUT = (float*)malloc(size * sizeof(float));

    const float inv_size = 1.0f / (float)size;
    const float factor   = beta;
    unsigned int k;
    for (k = 0; k < size; ++k){
      /* v in the range 0:(_width/2)^2 */
      const float v = ((float)k) * inv_size * cutoff2;
      (*LUT)[k] = kernel_value_CPU(factor * sqrtf(1.0f - (v / cutoff2)));
    }
    *sizeLUT = size;
  }
}

static inline float kernel_value_LUT(float v, float* LUT, int sizeLUT, float _1overCutoff2)
{
  /* v is already (distance^2), map it into LUT domain */
  const float scaled = v * (float)sizeLUT;
  const float t      = scaled * _1overCutoff2;
  unsigned int k0    = (unsigned int)t;
  if (k0 >= (unsigned int)(sizeLUT - 1)) {
    k0 = (unsigned int)(sizeLUT - 2);
  }
  const float v0 = (float)k0 / _1overCutoff2;
  const float lut0 = LUT[k0];
  const float lut1 = LUT[k0 + 1];
  return lut0 + ((scaled - v0) * (lut1 - lut0) * _1overCutoff2);
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
  float *dx2 = Dx2;
  float *dy2 = Dy2;
  float *dz2 = Dz2;

  float dy2dz2;
  float v;

  const unsigned int size_x = (unsigned int)params.gridSize[0];
  const unsigned int size_y = (unsigned int)params.gridSize[1];
  const unsigned int size_z = (unsigned int)params.gridSize[2];

  const float cutoff       = ((float)(params.kernelWidth)) * 0.5f;
  const float cutoff2      = cutoff * cutoff;
  const float _1overCutoff2 = 1.0f / cutoff2;

  const float oversample = params.oversample;
  const float kw         = params.kernelWidth;
  const float tmp        = (4.0f * kw * kw) / (oversample * oversample);
  const float os_m_half  = oversample - 0.5f;
  const float beta       = PI * sqrtf(tmp * (os_m_half * os_m_half) - 0.8f);

  const int useLUT = params.useLUT;

  unsigned int i;
  for (i = 0; i < n; i++){
    const ReconstructionSample pt = sample[i];

    const float kx = pt.kX;
    const float ky = pt.kY;
    const float kz = pt.kZ;

    NxL = (unsigned int)max(kx - cutoff, 0.0f);
    NxH = (unsigned int)min(kx + cutoff, (float)(size_x - 1U));

    NyL = (unsigned int)max(ky - cutoff, 0.0f);
    NyH = (unsigned int)min(ky + cutoff, (float)(size_y - 1U));

    NzL = (unsigned int)max(kz - cutoff, 0.0f);
    NzH = (unsigned int)min(kz + cutoff, (float)(size_z - 1U));

    if ((pt.real != 0.0f || pt.imag != 0.0f) && pt.sdc != 0.0f)
    {
      float *dz2_ptr = Dz2;
      for (nz = (int)NzL; (unsigned int)nz <= NzH; ++nz, ++dz2_ptr) {
        const float dz = kz - (float)nz;
        *dz2_ptr = dz * dz;
      }

      float *dx2_ptr = Dx2;
      for (nx = (int)NxL; (unsigned int)nx <= NxH; ++nx, ++dx2_ptr) {
        const float dx = kx - (float)nx;
        *dx2_ptr = dx * dx;
      }

      float *dy2_ptr = Dy2;
      for (ny = (int)NyL; (unsigned int)ny <= NyH; ++ny, ++dy2_ptr) {
        const float dy = ky - (float)ny;
        *dy2_ptr = dy * dy;
      }

      idxZ = (NzL - 1U) * size_x * size_y;
      dz2_ptr = Dz2;
      for (nz = (int)NzL; (unsigned int)nz <= NzH; ++nz, ++dz2_ptr)
      {
        idxZ += size_x * size_y;

        idxY = (NyL - 1U) * size_x;

        const float dz2_val = *dz2_ptr;
        if (dz2_val < cutoff2)
        {
          float *dy_local = Dy2;
          for (ny = (int)NyL; (unsigned int)ny <= NyH; ++ny, ++dy_local)
          {
            idxY += size_x;

            dy2dz2 = dz2_val + *dy_local;
            idx0   = idxY + idxZ;

            if (dy2dz2 < cutoff2)
            {
              float *dx_local = Dx2;
              for (nx = (int)NxL; (unsigned int)nx <= NxH; ++nx, ++dx_local)
              {
                v = dy2dz2 + *dx_local;

                if (v < cutoff2)
                {
                  idx = (unsigned int)nx + idx0;

                  if (useLUT){
                    w = kernel_value_LUT(v, LUT, (int)sizeLUT, _1overCutoff2) * pt.sdc;
                  } else {
                    w = kernel_value_CPU(beta * sqrtf(1.0f - (v * _1overCutoff2))) * pt.sdc;
                  }

                  gridData[idx].real += w * pt.real;
                  gridData[idx].imag += w * pt.imag;

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
