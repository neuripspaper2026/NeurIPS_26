#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "UDTypes.h"

#define max(x,y) ((x<y)?y:x)
#define min(x,y) ((x>y)?y:x)

#ifndef PI
#define PI 3.14159265359
#endif

/* Keep function signature and behavior identical */
float kernel_value_CPU(float v) {
  const float z = v * v;

  /* Horner-evaluated polynomial, coefficients unchanged, no temporaries */
  const float num =
    z * (z * (z * (z * (z * (z * (z * (z * (z * (z * (z * (z * (z *
    (z * 0.210580722890567e-22f  + 0.380715242345326e-19f) +
      0.479440257548300e-16f) + 0.435125971262668e-13f) +
      0.300931127112960e-10f) + 0.160224679395361e-7f)  +
      0.654858370096785e-5f)  + 0.202591084143397e-2f)  +
      0.463076284721000e0f)   + 0.754337328948189e2f)   +
      0.830792541809429e4f)   + 0.571661130563785e6f)   +
      0.216415572361227e8f)   + 0.356644482244025e9f)   +
      0.144048298227235e10f);

  const float den =
    z * (z * (z - 0.307646912682801e4f) + 0.347626332405882e7f)
      - 0.144048298227235e10f;

  return -num / den;
}

/* Keep function signature and behavior identical */
void calculateLUT(float beta, float width, float **LUT, unsigned int *sizeLUT) {
  const float cutoff2 = (width * width) * 0.25f;

  if (width > 0.0f) {
    /* compute size of LUT based on kernel width */
    const unsigned int size = (unsigned int)(10000.0f * width);

    /* allocate memory */
    float *lut = (float *)malloc(size * sizeof(float));
    *LUT = lut;

    const float invSize = 1.0f / (float)size;

    for (unsigned int k = 0; k < size; ++k) {
      /* v in the range 0:(_width/2)^2 */
      const float v = ((float)k) * invSize * cutoff2;

      lut[k] = kernel_value_CPU(beta * sqrtf(1.0f - (v / cutoff2)));
    }
    *sizeLUT = size;
  }
}

/* Keep function signature and behavior identical */
float kernel_value_LUT(float v, float *LUT, int sizeLUT, float _1overCutoff2) {
  /* Preload first coeffs to encourage autovectorization; keep math the same */
  float lut0 = LUT[0];
  float lut1 = LUT[1];
  (void)lut0;
  (void)lut1;

  /* v is scaled by sizeLUT */
  v *= (float)sizeLUT;

  const float t = v * _1overCutoff2;
  const unsigned int k0 = (unsigned int)t;
  const float v0 = (float)k0 / _1overCutoff2;

  const float y0 = LUT[k0];
  const float y1 = LUT[k0 + 1];

  return y0 + ((v - v0) * (y1 - y0) / _1overCutoff2);
}

/* Keep function signature and behavior identical */
int gridding_Gold(unsigned int n,
                  parameters params,
                  ReconstructionSample *sample,
                  float *LUT,
                  unsigned int sizeLUT,
                  cmplx *gridData,
                  float *sampleDensity) {
  const unsigned int size_x = (unsigned int)params.gridSize[0];
  const unsigned int size_y = (unsigned int)params.gridSize[1];
  const unsigned int size_z = (unsigned int)params.gridSize[2];

  /* cutoff radius and its square */
  const float cutoff = (float)params.kernelWidth * 0.5f;
  const float cutoff2 = cutoff * cutoff;
  const float invCutoff2 = 1.0f / cutoff2;

  /* precompute beta; unchanged formula */
  const float W = (float)params.kernelWidth;
  const float over = params.oversample;
  const float tmp = (4.0f * W * W) / (over * over) *
                    (over - 0.5f) * (over - 0.5f) - 0.8f;
  const float beta = PI * sqrtf(tmp);

  /* Pre-allocate maximum size buffers on stack as in original */
  float Dx2[100];
  float Dy2[100];
  float Dz2[100];

  for (unsigned int i = 0; i < n; ++i) {
    ReconstructionSample pt = sample[i];

    const float real = pt.real;
    const float imag = pt.imag;
    const float sdc  = pt.sdc;

    if ((real == 0.0f && imag == 0.0f) || sdc == 0.0f)
      continue;

    const float kx = pt.kX;
    const float ky = pt.kY;
    const float kz = pt.kZ;

    unsigned int NxL = (unsigned int)max(kx - cutoff, 0.0f);
    unsigned int NxH = (unsigned int)min(kx + cutoff, (float)(size_x - 1));

    unsigned int NyL = (unsigned int)max(ky - cutoff, 0.0f);
    unsigned int NyH = (unsigned int)min(ky + cutoff, (float)(size_y - 1));

    unsigned int NzL = (unsigned int)max(kz - cutoff, 0.0f);
    unsigned int NzH = (unsigned int)min(kz + cutoff, (float)(size_z - 1));

    /* Precompute squared distances for x, y, z */
    float *dx2 = Dx2;
    for (unsigned int nx = NxL; nx <= NxH; ++nx, ++dx2) {
      const float dx = kx - (float)nx;
      *dx2 = dx * dx;
    }

    float *dy2 = Dy2;
    for (unsigned int ny = NyL; ny <= NyH; ++ny, ++dy2) {
      const float dy = ky - (float)ny;
      *dy2 = dy * dy;
    }

    float *dz2 = Dz2;
    for (unsigned int nz = NzL; nz <= NzH; ++nz, ++dz2) {
      const float dz = kz - (float)nz;
      *dz2 = dz * dz;
    }

    /* Outer z-loop: maintain linear index directly; avoid recomputing */
    unsigned int idxZ = (NzL - 1U) * size_x * size_y;

    for (unsigned int iz = NzL, izLocal = 0; iz <= NzH; ++iz, ++izLocal) {
      idxZ += size_x * size_y;

      const float dz2_val = Dz2[izLocal];
      if (dz2_val >= cutoff2)
        continue;

      unsigned int idxY = (NyL - 1U) * size_x;

      for (unsigned int iy = NyL, iyLocal = 0; iy <= NyH; ++iy, ++iyLocal) {
        idxY += size_x;

        const float dy2_val = Dy2[iyLocal];
        const float dy2dz2 = dz2_val + dy2_val;

        if (dy2dz2 >= cutoff2)
          continue;

        const unsigned int idx0 = idxY + idxZ;

        /* Inner x-loop: index and kernel evaluation */
        for (unsigned int ix = NxL, ixLocal = 0; ix <= NxH; ++ix, ++ixLocal) {
          const float v = dy2dz2 + Dx2[ixLocal];

          if (v >= cutoff2)
            continue;

          const unsigned int idx = idx0 + ix;

          float w;
          if (params.useLUT) {
            w = kernel_value_LUT(v, LUT, (int)sizeLUT, invCutoff2) * sdc;
          } else {
            w = kernel_value_CPU(beta * sqrtf(1.0f - (v * invCutoff2))) * sdc;
          }

          gridData[idx].real += w * real;
          gridData[idx].imag += w * imag;
          sampleDensity[idx] += 1.0f;
        }
      }
    }
  }

  return 0;
}
