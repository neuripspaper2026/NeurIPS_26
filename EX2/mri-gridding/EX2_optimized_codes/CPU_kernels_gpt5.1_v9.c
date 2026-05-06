#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "UDTypes.h"

#define max(x,y) ((x<y)?y:x)
#define min(x,y) ((x>y)?y:x)

#define PI 3.14159265359

static inline float kernel_value_CPU(const float v)
{
  const float z = v * v;

  /* Unrolled Horner form for better performance and to help the compiler
   * generate efficient fused-multiply-add sequences. */
  float num =
    z * (0.210580722890567e-22f +
    z * (0.380715242345326e-19f +
    z * (0.479440257548300e-16f +
    z * (0.435125971262668e-13f +
    z * (0.300931127112960e-10f +
    z * (0.160224679395361e-7f  +
    z * (0.654858370096785e-5f  +
    z * (0.202591084143397e-2f  +
    z * (0.463076284721000e0f   +
    z * (0.754337328948189e2f   +
    z * (0.830792541809429e4f   +
    z * (0.571661130563785e6f   +
    z * (0.216415572361227e8f   +
    z * 0.356644482244025e9f))))))))))))));

  const float den = z * (z * (z - 0.307646912682801e4f) + 0.347626332405882e7f) - 0.144048298227235e10f;

  return -num / den;
}

void calculateLUT(float beta, float width, float **LUT, unsigned int *sizeLUT)
{
  const float cutoff2 = (width * width) * 0.25f;

  if (width > 0.0f) {
    /* compute size of LUT based on kernel width */
    const unsigned int size = (unsigned int)(10000.0f * width);

    /* allocate memory */
    float * restrict lut = (float *)malloc(size * sizeof(float));
    *LUT = lut;

    const float invSize = 1.0f / (float)size;

    unsigned int k;
    for (k = 0; k < size; ++k) {
      /* v in the range 0:(_width/2)^2 */
      const float v = ((float)k) * invSize * cutoff2;

      /* compute kernel value and store */
      lut[k] = kernel_value_CPU(beta * sqrtf(1.0f - (v / cutoff2)));
    }
    *sizeLUT = size;
  }
}

static inline float kernel_value_LUT(float v, const float * restrict LUT, int sizeLUT, const float _1overCutoff2)
{
  /* v is in [0, cutoff2]; map to LUT index domain where LUT is sampled
   * over [0, cutoff2] with sizeLUT entries. */
  const float scaled = v * (float)sizeLUT;
  unsigned int k0 = (unsigned int)(scaled * _1overCutoff2);

  if (__builtin_expect(k0 >= (unsigned int)(sizeLUT - 1), 0)) {
    k0 = (unsigned int)(sizeLUT - 2);
  }

  const float v0 = ((float)k0) / _1overCutoff2;
  const float dv = scaled - v0;

  const float f0 = LUT[k0];
  const float f1 = LUT[k0 + 1];

  return f0 + (dv * (f1 - f0) * _1overCutoff2);
}

int gridding_Gold(unsigned int n,
                  parameters params,
                  ReconstructionSample * restrict sample,
                  float * restrict LUT,
                  unsigned int sizeLUT,
                  cmplx * restrict gridData,
                  float * restrict sampleDensity)
{
  const unsigned int size_x = (unsigned int)params.gridSize[0];
  const unsigned int size_y = (unsigned int)params.gridSize[1];
  const unsigned int size_z = (unsigned int)params.gridSize[2];

  const float cutoff  = (float)params.kernelWidth * 0.5f;
  const float cutoff2 = cutoff * cutoff;
  const float _1overCutoff2 = 1.0f / cutoff2;

  const float os = params.oversample;
  const float kw = (float)params.kernelWidth;
  const float beta =
      PI * sqrtf(4.0f * kw * kw / (os * os) * (os - 0.5f) * (os - 0.5f) - 0.8f);

  /* The maximum kernel width determines the neighborhood size.
   * Use VLA sized with the maximum kernel width to avoid heap allocs. */
  const int maxKernelWidth = 100;
  float Dx2[maxKernelWidth];
  float Dy2[maxKernelWidth];
  float Dz2[maxKernelWidth];

  /* Parallelize over samples; each sample updates gridData and sampleDensity.
   * These arrays are shared; we must use atomic operations on updates. */
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(n, params, sample, LUT, sizeLUT, gridData, sampleDensity, size_x, size_y, size_z, cutoff, cutoff2, _1overCutoff2, beta, Dx2, Dy2, Dz2) schedule(static)
#endif
  for (int i = 0; i < (int)n; i++) {

    const ReconstructionSample pt = sample[i];

    const float kx = pt.kX;
    const float ky = pt.kY;
    const float kz = pt.kZ;

    /* Skip zero samples early */
    if (!((pt.real != 0.0f || pt.imag != 0.0f) && pt.sdc != 0.0f))
      continue;

    unsigned int NxL = (unsigned int)max(kx - cutoff, 0.0f);
    unsigned int NxH = (unsigned int)min(kx + cutoff, (float)(size_x - 1U));

    unsigned int NyL = (unsigned int)max(ky - cutoff, 0.0f);
    unsigned int NyH = (unsigned int)min(ky + cutoff, (float)(size_y - 1U));

    unsigned int NzL = (unsigned int)max(kz - cutoff, 0.0f);
    unsigned int NzH = (unsigned int)min(kz + cutoff, (float)(size_z - 1U));

    const int nxCount = (int)(NxH - NxL + 1U);
    const int nyCount = (int)(NyH - NyL + 1U);
    const int nzCount = (int)(NzH - NzL + 1U);

    if (nxCount <= 0 || nyCount <= 0 || nzCount <= 0)
      continue;

    /* Clamp to local buffer capacity */
    const int nxMax = nxCount > maxKernelWidth ? maxKernelWidth : nxCount;
    const int nyMax = nyCount > maxKernelWidth ? maxKernelWidth : nyCount;
    const int nzMax = nzCount > maxKernelWidth ? maxKernelWidth : nzCount;

    int nx;
    int ny;
    int nz;

    float * restrict dx2 = Dx2;
    for (nx = 0; nx < nxMax; ++nx, ++dx2) {
      const float x = (float)(NxL + (unsigned int)nx);
      const float diff = kx - x;
      *dx2 = diff * diff;
    }

    float * restrict dy2 = Dy2;
    for (ny = 0; ny < nyMax; ++ny, ++dy2) {
      const float y = (float)(NyL + (unsigned int)ny);
      const float diff = ky - y;
      *dy2 = diff * diff;
    }

    float * restrict dz2 = Dz2;
    for (nz = 0; nz < nzMax; ++nz, ++dz2) {
      const float z = (float)(NzL + (unsigned int)nz);
      const float diff = kz - z;
      *dz2 = diff * diff;
    }

    const unsigned int stride_xy = size_x * size_y;
    unsigned int baseZ = (NzL * stride_xy);

    for (nz = 0; nz < nzMax; ++nz) {
      const float dz2_val = Dz2[nz];

      if (dz2_val >= cutoff2) {
        baseZ += stride_xy;
        continue;
      }

      unsigned int baseY = NyL * size_x;
      const unsigned int idxZ = baseZ;

      for (ny = 0; ny < nyMax; ++ny) {
        const float dy2_val = Dy2[ny];
        const float dy2dz2 = dz2_val + dy2_val;

        if (dy2dz2 >= cutoff2) {
          baseY += size_x;
          continue;
        }

        const unsigned int idx0 = idxZ + baseY;

        for (nx = 0; nx < nxMax; ++nx) {
          const float v = dy2dz2 + Dx2[nx];

          if (v < cutoff2) {
            const unsigned int idx = (unsigned int)nx + NxL + idx0;

            float w;
            if (params.useLUT) {
              w = kernel_value_LUT(v, LUT, (int)sizeLUT, _1overCutoff2) * pt.sdc;
            } else {
              w = kernel_value_CPU(beta * sqrtf(1.0f - (v * _1overCutoff2))) * pt.sdc;
            }

#ifdef _OPENMP
#pragma omp atomic
#endif
            gridData[idx].real += (w * pt.real);
#ifdef _OPENMP
#pragma omp atomic
#endif
            gridData[idx].imag += (w * pt.imag);
#ifdef _OPENMP
#pragma omp atomic
#endif
            sampleDensity[idx] += 1.0f;
          }
        }

        baseY += size_x;
      }

      baseZ += stride_xy;
    }
  }

  return 0;
}
