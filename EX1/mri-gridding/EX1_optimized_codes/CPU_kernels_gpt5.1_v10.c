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

  /* Hoisted constant polynomial coefficients; no repeated construction */
  const float c0  = 0.210580722890567e-22f;
  const float c1  = 0.380715242345326e-19f;
  const float c2  = 0.479440257548300e-16f;
  const float c3  = 0.435125971262668e-13f;
  const float c4  = 0.300931127112960e-10f;
  const float c5  = 0.160224679395361e-7f;
  const float c6  = 0.654858370096785e-5f;
  const float c7  = 0.202591084143397e-2f;
  const float c8  = 0.463076284721000e0f;
  const float c9  = 0.754337328948189e2f;
  const float c10 = 0.830792541809429e4f;
  const float c11 = 0.571661130563785e6f;
  const float c12 = 0.216415572361227e8f;
  const float c13 = 0.356644482244025e9f;
  const float c14 = 0.144048298227235e10f;

  /* Horner scheme for numerator */
  float num =
    ((((((((((((((c0  * z + c1)  * z + c2)  * z + c3)  * z + c4)  * z +
                c5)  * z + c6)  * z + c7)  * z + c8)  * z + c9)  * z +
              c10) * z + c11) * z + c12) * z + c13) * z + c14;

  /* Denominator polynomial */
  const float d0 = -0.307646912682801e4f;
  const float d1 =  0.347626332405882e7f;
  const float d2 = -0.144048298227235e10f;

  float den = z * (z * (z + d0) + d1) + d2;

  return -num / den;
}

void calculateLUT(float beta, float width, float** LUT, unsigned int* sizeLUT){
  float v;
  const float cutoff2 = (width * width) * 0.25f;

  unsigned int size;

  if (width > 0.0f){
    /* size of LUT based on kernel width */
    size = (unsigned int)(10000.0f * width);

    /* allocate memory */
    (*LUT) = (float*)malloc(size * sizeof(float));

    const float invSize = 1.0f / (float)size;
    unsigned int k;
    for (k = 0; k < size; ++k){
      /* v in the range 0:(_width/2)^2 */
      v = ((float)k) * invSize * cutoff2;

      /* compute kernel value and store */
      (*LUT)[k] = kernel_value_CPU(beta * sqrtf(1.0f - (v / cutoff2)));
    }
    (*sizeLUT) = size;
  }
}

static inline float kernel_value_LUT(float v, const float* LUT, int sizeLUT, float _1overCutoff2)
{
  /* v is already squared distance; scale once */
  v *= (float)sizeLUT;
  const float vk = v * _1overCutoff2;
  unsigned int k0 = (unsigned int)vk;

  /* Clamp to ensure k0+1 in range */
  if (k0 >= (unsigned int)(sizeLUT - 1)) {
    k0 = (unsigned int)(sizeLUT - 2);
  }

  const float v0 = (float)k0 / _1overCutoff2;
  const float dv = v - v0;
  const float y0 = LUT[k0];
  const float y1 = LUT[k0 + 1];

  return y0 + dv * (y1 - y0) * (1.0f / _1overCutoff2);
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
  float *dx2 = NULL;
  float *dy2 = NULL;
  float *dz2 = NULL;

  float dy2dz2;
  float v;

  const unsigned int size_x = (unsigned int)params.gridSize[0];
  const unsigned int size_y = (unsigned int)params.gridSize[1];
  const unsigned int size_z = (unsigned int)params.gridSize[2];

  const float cutoff  = ((float)(params.kernelWidth)) * 0.5f; /* cutoff radius */
  const float cutoff2 = cutoff * cutoff;                      /* square of cutoff radius */
  const float _1overCutoff2 = 1.0f / cutoff2;                 /* 1 over square of cutoff radius */

  const float kw = (float)params.kernelWidth;
  const float os = params.oversample;
  const float os_minus_half = os - 0.5f;
  const float beta_factor = 4.0f * kw * kw / (os * os);
  const float beta =
      (float)PI * sqrtf(beta_factor * os_minus_half * os_minus_half - 0.8f);

  const int useLUT = params.useLUT;
  const float *const LUT_const = LUT;
  const int sizeLUT_int = (int)sizeLUT;

  const unsigned int plane_size = size_x * size_y;

  unsigned int i;
  for (i = 0; i < n; i++){
    const ReconstructionSample pt = sample[i];

    const float kx = pt.kX;
    const float ky = pt.kY;
    const float kz = pt.kZ;

    /* Pre-check to avoid useless work for many empty samples */
    if ((pt.real == 0.0f && pt.imag == 0.0f) || pt.sdc == 0.0f) {
      continue;
    }

    const float kx_minus_cut = kx - cutoff;
    const float kx_plus_cut  = kx + cutoff;
    const float ky_minus_cut = ky - cutoff;
    const float ky_plus_cut  = ky + cutoff;
    const float kz_minus_cut = kz - cutoff;
    const float kz_plus_cut  = kz + cutoff;

    NxL = (unsigned int)max(kx_minus_cut, 0.0f);
    NxH = (unsigned int)min(kx_plus_cut, (float)(size_x - 1u));

    NyL = (unsigned int)max(ky_minus_cut, 0.0f);
    NyH = (unsigned int)min(ky_plus_cut, (float)(size_y - 1u));

    NzL = (unsigned int)max(kz_minus_cut, 0.0f);
    NzH = (unsigned int)min(kz_plus_cut, (float)(size_z - 1u));

    const int NzH_int = (int)NzH;
    const int NzL_int = (int)NzL;
    const int NyH_int = (int)NyH;
    const int NyL_int = (int)NyL;
    const int NxH_int = (int)NxH;
    const int NxL_int = (int)NxL;

    /* precompute squared distances */
    dz2 = Dz2;
    for (nz = NzL_int; nz <= NzH_int; ++nz, ++dz2){
      const float dz = kz - (float)nz;
      *dz2 = dz * dz;
    }
    dx2 = Dx2;
    for (nx = NxL_int; nx <= NxH_int; ++nx, ++dx2){
      const float dx = kx - (float)nx;
      *dx2 = dx * dx;
    }
    dy2 = Dy2;
    for (ny = NyL_int; ny <= NyH_int; ++ny, ++dy2){
      const float dy = ky - (float)ny;
      *dy2 = dy * dy;
    }

    idxZ = (NzL - 1u) * plane_size;
    dz2 = Dz2;
    for (nz = NzL_int; nz <= NzH_int; ++nz, ++dz2)
    {
      idxZ += plane_size;

      idxY = (NyL - 1u) * size_x;

      /* loop over y indexes, but only if z-distance is close enough */
      if (*dz2 < cutoff2)
      {
        dy2 = Dy2;
        for (ny = NyL_int; ny <= NyH_int; ++ny, ++dy2)
        {
          idxY += size_x;

          dy2dz2 = (*dz2) + (*dy2);

          if (dy2dz2 < cutoff2)
          {
            idx0 = idxY + idxZ;

            dx2 = Dx2;
            for (nx = NxL_int; nx <= NxH_int; ++nx, ++dx2)
            {
              v = dy2dz2 + (*dx2);

              if (v < cutoff2)
              {
                idx = (unsigned int)nx + idx0;

                if (useLUT){
                  w = kernel_value_LUT(v, LUT_const, sizeLUT_int, _1overCutoff2) * pt.sdc;
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

  return 0;
}
