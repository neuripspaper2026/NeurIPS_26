#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "UDTypes.h"

#define max(x,y) ((x)<(y)?(y):(x))
#define min(x,y) ((x)>(y)?(y):(x))

#define PI 3.14159265359

static inline float kernel_value_CPU(float v){

  const float z = v * v;

  /* Horner-evaluated polynomial (reassociated for fewer temporaries) */
  float num =
    z * (z * (z * (z * (z * (z * (z * (z * (z * (z * (z * (z * (z *
    (z * 0.210580722890567e-22f  + 0.380715242345326e-19f) +
         0.479440257548300e-16f) + 0.435125971262668e-13f) +
         0.300931127112960e-10f) + 0.160224679395361e-7f) +
         0.654858370096785e-5f) + 0.202591084143397e-2f) +
         0.463076284721000e0f)  + 0.754337328948189e2f) +
         0.830792541809429e4f)  + 0.571661130563785e6f) +
         0.216415572361227e8f)  + 0.356644482244025e9f) +
         0.144048298227235e10f);

  float den = z * (z * (z - 0.307646912682801e4f) + 0.347626332405882e7f)
              - 0.144048298227235e10f;

  return -num / den;
}

void calculateLUT(float beta, float width, float** LUT, unsigned int* sizeLUT){
  if (width <= 0.0f) {
    *LUT = NULL;
    *sizeLUT = 0;
    return;
  }

  const float cutoff2 = (width * width) * 0.25f;

  /* compute size of LUT based on kernel width */
  const unsigned int size = (unsigned int)(10000.0f * width);

  /* allocate memory */
  float *restrict lut_local = (float *)malloc(size * sizeof(float));
  if (!lut_local) {
    *LUT = NULL;
    *sizeLUT = 0;
    return;
  }

  const float inv_size = 1.0f / (float)size;
  const float inv_cutoff2 = 1.0f / cutoff2;

  for (unsigned int k = 0; k < size; ++k){
    /* v in the range 0:(_width/2)^2 */
    const float t = (float)k * inv_size;
    const float v = t * cutoff2;

    /* compute kernel value and store */
    const float arg = beta * sqrtf(fmaxf(0.0f, 1.0f - v * inv_cutoff2));
    lut_local[k] = kernel_value_CPU(arg);
  }

  *LUT = lut_local;
  *sizeLUT = size;
}

static inline float kernel_value_LUT(float v, const float* restrict LUT,
                                     unsigned int sizeLUT,
                                     float _1overCutoff2)
{
  /* map v from [0,cutoff2] into [0,sizeLUT) index space */
  const float scaled = v * (float)sizeLUT * _1overCutoff2;
  unsigned int k0 = (unsigned int)scaled;

  if (k0 >= sizeLUT - 1U) {
    k0 = sizeLUT - 2U;
  }

  const float v0 = (float)k0 / _1overCutoff2;
  const float dv = scaled - (float)k0;
  const float y0 = LUT[k0];
  const float y1 = LUT[k0 + 1U];

  return y0 + dv * (y1 - y0);
}

int gridding_Gold(unsigned int n,
                  parameters params,
                  ReconstructionSample* restrict sample,
                  float* restrict LUT,
                  unsigned int sizeLUT,
                  cmplx* restrict gridData,
                  float* restrict sampleDensity){

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

  float *restrict dx2 = NULL;
  float *restrict dy2 = NULL;
  float *restrict dz2 = NULL;

  float dy2dz2;
  float v;

  const unsigned int size_x = (unsigned int)params.gridSize[0];
  const unsigned int size_y = (unsigned int)params.gridSize[1];
  const unsigned int size_z = (unsigned int)params.gridSize[2];

  const float cutoff  = 0.5f * (float)params.kernelWidth;  /* cutoff radius      */
  const float cutoff2 = cutoff * cutoff;                   /* square of cutoff   */
  const float _1overCutoff2 = 1.0f / cutoff2;              /* 1 / cutoff^2       */

  const float kw = (float)params.kernelWidth;
  const float os = params.oversample;
  const float t1 = 4.0f * kw * kw / (os * os);
  const float t2 = (os - 0.5f);
  const float beta = PI * sqrtf(t1 * t2 * t2 - 0.8f);

  for (unsigned int i = 0; i < n; ++i){
    const ReconstructionSample pt = sample[i];

    const float kx = pt.kX;
    const float ky = pt.kY;
    const float kz = pt.kZ;

    if ((pt.real == 0.0f && pt.imag == 0.0f) || pt.sdc == 0.0f)
      continue;

    /* precompute integer bounds (clamped) */
    const float fxL = kx - cutoff;
    const float fxH = kx + cutoff;
    const float fyL = ky - cutoff;
    const float fyH = ky + cutoff;
    const float fzL = kz - cutoff;
    const float fzH = kz + cutoff;

    NxL = (unsigned int)max(fxL, 0.0f);
    NxH = (unsigned int)min(fxH, (float)(size_x - 1U));
    NyL = (unsigned int)max(fyL, 0.0f);
    NyH = (unsigned int)min(fyH, (float)(size_y - 1U));
    NzL = (unsigned int)max(fzL, 0.0f);
    NzH = (unsigned int)min(fzH, (float)(size_z - 1U));

    const unsigned int countX = NxH - NxL + 1U;
    const unsigned int countY = NyH - NyL + 1U;
    const unsigned int countZ = NzH - NzL + 1U;

    /* precompute squared distances along axes */
    dz2 = Dz2;
    for (nz = (int)NzL; (unsigned int)nz <= NzH; ++nz, ++dz2) {
      const float dz = kz - (float)nz;
      *dz2 = dz * dz;
    }

    dx2 = Dx2;
    for (nx = (int)NxL; (unsigned int)nx <= NxH; ++nx, ++dx2) {
      const float dx = kx - (float)nx;
      *dx2 = dx * dx;
    }

    dy2 = Dy2;
    for (ny = (int)NyL; (unsigned int)ny <= NyH; ++ny, ++dy2) {
      const float dy = ky - (float)ny;
      *dy2 = dy * dy;
    }

    const unsigned int planeStride = size_x * size_y;
    const float sdc = pt.sdc;
    const float pre_real = pt.real;
    const float pre_imag = pt.imag;

    idxZ = (NzL - 1U) * planeStride;
    dz2 = Dz2;

    for (unsigned int iz = 0; iz < countZ; ++iz, ++dz2) {

      idxZ += planeStride;

      if (*dz2 >= cutoff2)
        continue;

      idxY = (NyL - 1U) * size_x;
      dy2 = Dy2;

      for (unsigned int iy = 0; iy < countY; ++iy, ++dy2) {

        idxY += size_x;

        dy2dz2 = *dz2 + *dy2;

        if (dy2dz2 >= cutoff2)
          continue;

        idx0 = idxY + idxZ;

        dx2 = Dx2;
        for (unsigned int ix = 0; ix < countX; ++ix, ++dx2) {

          v = dy2dz2 + *dx2;

          if (v >= cutoff2)
            continue;

          idx = (unsigned int)NxL + ix + idx0;

          if (params.useLUT) {
            w = kernel_value_LUT(v, LUT, sizeLUT, _1overCutoff2) * sdc;
          } else {
            const float arg = beta * sqrtf(fmaxf(0.0f, 1.0f - v * _1overCutoff2));
            w = kernel_value_CPU(arg) * sdc;
          }

          gridData[idx].real += w * pre_real;
          gridData[idx].imag += w * pre_imag;

          sampleDensity[idx] += 1.0f;
        }
      }
    }
  }

  return 0;
}
