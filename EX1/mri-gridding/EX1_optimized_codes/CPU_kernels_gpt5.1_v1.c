#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "UDTypes.h"

#define max(x,y) ((x<y)?y:x)
#define min(x,y) ((x>y)?y:x)

#define PI 3.14159265359

/* Helper macros for Horner evaluation of the numerator polynomial */
#define HORNER_STEP(z, c) (z * (c))

float kernel_value_CPU(float v) {

  const float z = v * v;

  /* Unrolled Horner scheme for numerator polynomial, using fewer temporaries
     and avoiding deep nested parentheses. */
  float num;

  num  = 0.210580722890567e-22f;
  num  = HORNER_STEP(z, num) + 0.380715242345326e-19f;
  num  = HORNER_STEP(z, num) + 0.479440257548300e-16f;
  num  = HORNER_STEP(z, num) + 0.435125971262668e-13f;
  num  = HORNER_STEP(z, num) + 0.300931127112960e-10f;
  num  = HORNER_STEP(z, num) + 0.160224679395361e-7f;
  num  = HORNER_STEP(z, num) + 0.654858370096785e-5f;
  num  = HORNER_STEP(z, num) + 0.202591084143397e-2f;
  num  = HORNER_STEP(z, num) + 0.463076284721000e0f;
  num  = HORNER_STEP(z, num) + 0.754337328948189e2f;
  num  = HORNER_STEP(z, num) + 0.830792541809429e4f;
  num  = HORNER_STEP(z, num) + 0.571661130563785e6f;
  num  = HORNER_STEP(z, num) + 0.216415572361227e8f;
  num  = HORNER_STEP(z, num) + 0.356644482244025e9f;
  num  = HORNER_STEP(z, num) + 0.144048298227235e10f;

  const float den =
      z * (z * (z - 0.307646912682801e4f) + 0.347626332405882e7f)
      - 0.144048298227235e10f;

  return -num / den;
}

#undef HORNER_STEP

void calculateLUT(float beta, float width, float **LUT, unsigned int *sizeLUT) {
  if (width <= 0.0f) {
    return;
  }

  const float cutoff2 = (width * width) * 0.25f;

  /* compute size of LUT based on kernel width */
  const unsigned int size = (unsigned int)(10000.0f * width);

  /* allocate memory */
  float *lutLocal = (float *)malloc(size * sizeof(float));
  if (!lutLocal) {
    *LUT = NULL;
    *sizeLUT = 0;
    return;
  }

  const float invSize = 1.0f / (float)size;

  unsigned int k;
  for (k = 0; k < size; ++k) {
    /* v in the range 0 : (width/2)^2 */
    const float v = ((float)k) * invSize * cutoff2;
    lutLocal[k] = kernel_value_CPU(beta * sqrtf(1.0f - (v / cutoff2)));
  }

  *LUT = lutLocal;
  *sizeLUT = size;
}

float kernel_value_LUT(float v, float *LUT, int sizeLUT, float _1overCutoff2) {
  /* v is expected in [0, cutoff2]; here v is scaled outside before call. */
  const float scaled = v * (float)sizeLUT;
  const unsigned int k0 = (unsigned int)(scaled * _1overCutoff2);
  const float v0 = ((float)k0) / _1overCutoff2;
  const float base = LUT[k0];
  return base + ((scaled - v0) * (LUT[k0 + 1] - base) / _1overCutoff2);
}

int gridding_Gold(unsigned int n, parameters params, ReconstructionSample *sample,
                  float *LUT, unsigned int sizeLUT, cmplx *gridData,
                  float *sampleDensity) {

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

  const float cutoff = ((float)(params.kernelWidth)) * 0.5f;
  const float cutoff2 = cutoff * cutoff;
  const float _1overCutoff2 = 1.0f / cutoff2;

  const float kw = (float)params.kernelWidth;
  const float os = params.oversample;
  const float os_m_5 = os - 0.5f;
  const float factor = (4.0f * kw * kw) / (os * os) * (os_m_5 * os_m_5) - 0.8f;
  const float beta = PI * sqrtf(factor);

  const int useLUT = params.useLUT;
  const float lutScale = (float)sizeLUT;

  unsigned int i;
  for (i = 0; i < n; i++) {
    ReconstructionSample pt = sample[i];

    const float kx = pt.kX;
    const float ky = pt.kY;
    const float kz = pt.kZ;

    /* pre-check to avoid useless work when sample is zero or sdc is zero */
    if (((pt.real == 0.0f) && (pt.imag == 0.0f)) || pt.sdc == 0.0f) {
      continue;
    }

    /* precompute integer bounds (clamped) */
    const float cutoffNeg = -cutoff;
    const float cutoffPos = cutoff;
    const float kxMin = kx + cutoffNeg;
    const float kxMax = kx + cutoffPos;
    const float kyMin = ky + cutoffNeg;
    const float kyMax = ky + cutoffPos;
    const float kzMin = kz + cutoffNeg;
    const float kzMax = kz + cutoffPos;

    if (kxMax < 0.0f || kxMin > (float)(size_x - 1) ||
        kyMax < 0.0f || kyMin > (float)(size_y - 1) ||
        kzMax < 0.0f || kzMin > (float)(size_z - 1)) {
      continue;
    }

    NxL = (unsigned int)max(kxMin, 0.0f);
    NxH = (unsigned int)min(kxMax, (float)(size_x - 1));

    NyL = (unsigned int)max(kyMin, 0.0f);
    NyH = (unsigned int)min(kyMax, (float)(size_y - 1));

    NzL = (unsigned int)max(kzMin, 0.0f);
    NzH = (unsigned int)min(kzMax, (float)(size_z - 1));

    const unsigned int countX = NxH - NxL + 1;
    const unsigned int countY = NyH - NyL + 1;
    const unsigned int countZ = NzH - NzL + 1;

    if (countX > 100u || countY > 100u || countZ > 100u) {
      /* Fallback to original access pattern if bounds exceed static buffers */
      unsigned int nz_fallback, ny_fallback, nx_fallback;
      for (nz_fallback = NzL; nz_fallback <= NzH; ++nz_fallback) {
        const float dz2_f = (kz - (float)nz_fallback) * (kz - (float)nz_fallback);
        if (dz2_f >= cutoff2) continue;

        const unsigned int zBase = nz_fallback * size_x * size_y;
        for (ny_fallback = NyL; ny_fallback <= NyH; ++ny_fallback) {
          const float dy2_f = (ky - (float)ny_fallback) * (ky - (float)ny_fallback);
          dy2dz2 = dz2_f + dy2_f;
          if (dy2dz2 >= cutoff2) continue;

          const unsigned int yBase = ny_fallback * size_x;
          idx0 = zBase + yBase;

          for (nx_fallback = NxL; nx_fallback <= NxH; ++nx_fallback) {
            const float dx2_f = (kx - (float)nx_fallback) * (kx - (float)nx_fallback);
            v = dy2dz2 + dx2_f;
            if (v >= cutoff2) continue;

            idx = nx_fallback + idx0;

            if (useLUT) {
              const float scaled = v * lutScale;
              const unsigned int k0 = (unsigned int)(scaled * _1overCutoff2);
              const float v0 = ((float)k0) / _1overCutoff2;
              const float base = LUT[k0];
              w = (base + ((scaled - v0) * (LUT[k0 + 1] - base) / _1overCutoff2)) * pt.sdc;
            } else {
              w = kernel_value_CPU(beta * sqrtf(1.0f - (v * _1overCutoff2))) * pt.sdc;
            }

            gridData[idx].real += (w * pt.real);
            gridData[idx].imag += (w * pt.imag);
            sampleDensity[idx] += 1.0f;
          }
        }
      }
      continue;
    }

    /* Precompute squared distance components along each axis */
    float *restrict dx2Base = Dx2;
    float *restrict dy2Base = Dy2;
    float *restrict dz2Base = Dz2;

    for (dz2 = dz2Base, nz = (int)NzL; (unsigned int)nz <= NzH; ++nz, ++dz2) {
      const float diffz = kz - (float)nz;
      *dz2 = diffz * diffz;
    }
    for (dx2 = dx2Base, nx = (int)NxL; (unsigned int)nx <= NxH; ++nx, ++dx2) {
      const float diffx = kx - (float)nx;
      *dx2 = diffx * diffx;
    }
    for (dy2 = dy2Base, ny = (int)NyL; (unsigned int)ny <= NyH; ++ny, ++dy2) {
      const float diffy = ky - (float)ny;
      *dy2 = diffy * diffy;
    }

    /* Base Z offset starts before the first slice; increment at loop start */
    idxZ = (NzL - 1u) * size_x * size_y;
    dz2 = dz2Base;

    const float ptReal = pt.real;
    const float ptImag = pt.imag;
    const float sdc = pt.sdc;

    for (nz = (int)NzL; (unsigned int)nz <= NzH; ++nz, ++dz2) {

      idxZ += size_x * size_y;
      const float dz2val = *dz2;
      if (dz2val >= cutoff2) {
        continue;
      }

      idxY = (NyL - 1u) * size_x;
      dy2 = dy2Base;

      for (ny = (int)NyL; (unsigned int)ny <= NyH; ++ny, ++dy2) {

        idxY += size_x;
        dy2dz2 = dz2val + *dy2;
        if (dy2dz2 >= cutoff2) {
          continue;
        }

        idx0 = idxY + idxZ;
        dx2 = dx2Base;

        for (nx = (int)NxL; (unsigned int)nx <= NxH; ++nx, ++dx2) {

          v = dy2dz2 + *dx2;
          if (v >= cutoff2) {
            continue;
          }

          idx = (unsigned int)nx + idx0;

          if (useLUT) {
            const float scaled = v * lutScale;
            const unsigned int k0 = (unsigned int)(scaled * _1overCutoff2);
            const float v0 = ((float)k0) / _1overCutoff2;
            const float base = LUT[k0];
            w = (base + ((scaled - v0) * (LUT[k0 + 1] - base) / _1overCutoff2)) * sdc;
          } else {
            w = kernel_value_CPU(beta * sqrtf(1.0f - (v * _1overCutoff2))) * sdc;
          }

          gridData[idx].real += (w * ptReal);
          gridData[idx].imag += (w * ptImag);
          sampleDensity[idx] += 1.0f;
        }
      }
    }
  }

  return 0;
}
