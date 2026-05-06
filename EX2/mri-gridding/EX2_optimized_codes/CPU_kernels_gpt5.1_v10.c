#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "UDTypes.h"

#define max(x,y) ((x<y)?y:x)
#define min(x,y) ((x>y)?y:x)

#define PI 3.14159265359

static inline float kernel_value_CPU(const float v) {
  const float z = v * v;

  /* Horner evaluation of the numerator polynomial */
  float num =
    z * (z * (z * (z * (z * (z * (z * (z * (z * (z * (z * (z * (z *
    (z * 0.210580722890567e-22f  + 0.380715242345326e-19f) +
         0.479440257548300e-16f) + 0.435125971262668e-13f) +
         0.300931127112960e-10f) + 0.160224679395361e-7f ) +
         0.654858370096785e-5f)  + 0.202591084143397e-2f) +
         0.463076284721000e0f)   + 0.754337328948189e2f) +
         0.830792541809429e4f)   + 0.571661130563785e6f) +
         0.216415572361227e8f)   + 0.356644482244025e9f) +
         0.144048298227235e10f);

  /* Denominator polynomial */
  const float den =
    z * (z * (z - 0.307646912682801e4f) + 0.347626332405882e7f)
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
  const unsigned int size = (unsigned int)(10000.0f * width);

  float *localLUT = (float*) malloc(size * sizeof(float));
  if (!localLUT) {
    *LUT = NULL;
    *sizeLUT = 0;
    return;
  }

  /* Precompute reciprocal once */
  const float inv_size = 1.0f / (float)size;

  unsigned int k;
  for (k = 0; k < size; ++k) {
    const float t = (float)k * inv_size;      /* t in [0,1) */
    const float v = t * cutoff2;              /* v in [0, cutoff2) */
    /* Argument of kernel_value_CPU is always in [0, beta] */
    localLUT[k] = kernel_value_CPU(beta * sqrtf(1.0f - (v / cutoff2)));
  }

  *LUT = localLUT;
  *sizeLUT = size;
}

static inline float kernel_value_LUT(float v, const float* __restrict LUT,
                                     const int sizeLUT,
                                     const float _1overCutoff2)
{
  /* Map v in [0, cutoff2] to LUT index space [0, sizeLUT] */
  const float scaled = v * (float)sizeLUT;
  const float scaled_idx = scaled * _1overCutoff2;
  unsigned int k0 = (unsigned int)scaled_idx;

  if (k0 >= (unsigned int)(sizeLUT - 1)) {
    /* Clamp to last interval to avoid out-of-bounds */
    k0 = (unsigned int)(sizeLUT - 2);
  }

  const float v0 = (float)k0 / _1overCutoff2;
  const float y0 = LUT[k0];
  const float y1 = LUT[k0 + 1];
  return y0 + ((scaled - v0) * (y1 - y0) * _1overCutoff2);
}

int gridding_Gold(unsigned int n,
                  parameters params,
                  ReconstructionSample* __restrict sample,
                  float* __restrict LUT,
                  unsigned int sizeLUT,
                  cmplx* __restrict gridData,
                  float* __restrict sampleDensity)
{
  const unsigned int size_x = (unsigned int)params.gridSize[0];
  const unsigned int size_y = (unsigned int)params.gridSize[1];
  const unsigned int size_z = (unsigned int)params.gridSize[2];

  const float cutoff  = ((float)(params.kernelWidth)) * 0.5f;   /* cutoff radius */
  const float cutoff2 = cutoff * cutoff;                        /* square of cutoff radius */
  const float _1overCutoff2 = 1.0f / cutoff2;

  const float oversample = params.oversample;
  const float kernelWidth = (float)params.kernelWidth;
  const float t1 = 4.0f * kernelWidth * kernelWidth /
                   (oversample * oversample);
  const float t2 = (oversample - 0.5f) * (oversample - 0.5f) - 0.8f;
  const float beta = PI * sqrtf(t1 * t2);

  /* Stack-allocated distance buffers; kernelWidth is small in practice */
  float Dx2[100];
  float Dy2[100];
  float Dz2[100];

  unsigned int i;

#pragma omp parallel for private(i) schedule(static) if(n > 16)
  for (i = 0; i < n; i++){
    ReconstructionSample pt = sample[i];

    const float pt_real = pt.real;
    const float pt_imag = pt.imag;
    const float pt_sdc  = pt.sdc;

    if ((pt_real == 0.0f && pt_imag == 0.0f) || pt_sdc == 0.0f) {
      continue;
    }

    const float kx = pt.kX;
    const float ky = pt.kY;
    const float kz = pt.kZ;

    /* Compute integer bounds; grid indices are [0, size_* - 1] */
    unsigned int NxL = (unsigned int)max(kx - cutoff, 0.0f);
    unsigned int NxH = (unsigned int)min(kx + cutoff, (float)(size_x - 1));

    unsigned int NyL = (unsigned int)max(ky - cutoff, 0.0f);
    unsigned int NyH = (unsigned int)min(ky + cutoff, (float)(size_y - 1));

    unsigned int NzL = (unsigned int)max(kz - cutoff, 0.0f);
    unsigned int NzH = (unsigned int)min(kz + cutoff, (float)(size_z - 1));

    int nx, ny, nz;

    float *dx2;
    float *dy2;
    float *dz2;

    /* Precompute squared distances along each axis in the local support */
    for (dz2 = Dz2, nz = (int)NzL; (unsigned int)nz <= NzH; ++nz, ++dz2) {
      const float diff = kz - (float)nz;
      *dz2 = diff * diff;
    }
    for (dx2 = Dx2, nx = (int)NxL; (unsigned int)nx <= NxH; ++nx, ++dx2) {
      const float diff = kx - (float)nx;
      *dx2 = diff * diff;
    }
    for (dy2 = Dy2, ny = (int)NyL; (unsigned int)ny <= NyH; ++ny, ++dy2) {
      const float diff = ky - (float)ny;
      *dy2 = diff * diff;
    }

    unsigned int idxZ = (NzL - 1U) * size_x * size_y;

    for (dz2 = Dz2, nz = (int)NzL; (unsigned int)nz <= NzH; ++nz, ++dz2) {
      idxZ += size_x * size_y; /* move to next z-slab */

      const float dz2_val = *dz2;
      if (dz2_val >= cutoff2) {
        continue;
      }

      unsigned int idxY = (NyL - 1U) * size_x;

      for (dy2 = Dy2, ny = (int)NyL; (unsigned int)ny <= NyH; ++ny, ++dy2) {
        idxY += size_x; /* move to next y-row */

        const float dy2_val = *dy2;
        const float dy2dz2 = dz2_val + dy2_val;
        if (dy2dz2 >= cutoff2) {
          continue;
        }

        const unsigned int idx0 = idxY + idxZ;

        for (dx2 = Dx2, nx = (int)NxL; (unsigned int)nx <= NxH; ++nx, ++dx2) {
          const float v = dy2dz2 + *dx2;

          if (v < cutoff2) {
            const unsigned int idx = (unsigned int)nx + idx0;

            float w;
            if (params.useLUT) {
              w = kernel_value_LUT(v, LUT, (int)sizeLUT, _1overCutoff2) * pt_sdc;
            } else {
              w = kernel_value_CPU(beta * sqrtf(1.0f - (v * _1overCutoff2))) * pt_sdc;
            }

            /* Update shared grid; ensure correctness under OpenMP */
#pragma omp atomic
            gridData[idx].real += w * pt_real;
#pragma omp atomic
            gridData[idx].imag += w * pt_imag;
#pragma omp atomic
            sampleDensity[idx] += 1.0f;
          }
        }
      }
    }
  }

  return 0;
}
