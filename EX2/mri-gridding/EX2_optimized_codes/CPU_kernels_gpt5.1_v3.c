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

static inline float kernel_value_CPU(float v){

  const float z = v*v;

  /* Horner-evaluated polynomial with fused structure kept as-is for numerical identity */
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

  const float den =
    z * (z * (z - 0.307646912682801e4f) + 0.347626332405882e7f)
      - 0.144048298227235e10f;

  return -num/den;
}

void calculateLUT(float beta, float width, float** LUT, unsigned int* sizeLUT){
  if (width <= 0.0f) {
    return;
  }

  const float cutoff2 = (width*width)*0.25f;
  const unsigned int size = (unsigned int)(10000.0f*width);

  float *localLUT = (float*) malloc(size * sizeof(float));
  if (!localLUT) {
    *LUT = NULL;
    *sizeLUT = 0;
    return;
  }

  const float invSize = 1.0f / (float)size;

  unsigned int k;
  for(k = 0; k < size; ++k){
    const float v = ((float)k) * invSize * cutoff2;
    localLUT[k] = kernel_value_CPU(beta*sqrtf(1.0f-(v/cutoff2)));
  }

  *LUT = localLUT;
  *sizeLUT = size;
}

static inline float kernel_value_LUT(float v, const float* __restrict LUT, int sizeLUT, float _1overCutoff2)
{
  /* v is already squared distance; scale by table size once */
  const float scaled = v * (float)sizeLUT;
  const float scaledDiv = scaled * _1overCutoff2;

  unsigned int k0 = (unsigned int)scaledDiv;

  if (k0 >= (unsigned int)(sizeLUT-1)) {
    k0 = (unsigned int)(sizeLUT-2);
  }

  const float v0 = ((float)k0) / _1overCutoff2;
  return  LUT[k0] + ((scaled - v0) * (LUT[k0+1] - LUT[k0]) * _1overCutoff2);
}

int gridding_Gold(unsigned int n,
                  parameters params,
                  ReconstructionSample* __restrict sample,
                  float* __restrict LUT,
                  unsigned int sizeLUT,
                  cmplx* __restrict gridData,
                  float* __restrict sampleDensity){

  const unsigned int size_x = (unsigned int)params.gridSize[0];
  const unsigned int size_y = (unsigned int)params.gridSize[1];
  const unsigned int size_z = (unsigned int)params.gridSize[2];

  const float cutoff   = ((float)(params.kernelWidth)) * 0.5f;     /* cutoff radius */
  const float cutoff2  = cutoff * cutoff;                          /* cutoff radius squared */
  const float invCut2  = 1.0f / cutoff2;                           /* 1 / cutoff^2 */

  const float oversample = params.oversample;
  const float kw = (float)params.kernelWidth;
  const float beta = PI * sqrtf(
      (4.0f * kw * kw) / (oversample * oversample) *
      (oversample - 0.5f) * (oversample - 0.5f) - 0.8f);

  /* Precompute strides to avoid repeated multiplies in inner loops */
  const unsigned int planeStride = size_x * size_y;
  const unsigned int rowStride   = size_x;

  unsigned int i;

  /* Parallelize across input samples. Each sample writes to arbitrary grid locations,
   * so use OpenMP atomics to protect updates. The amount of work per sample is large
   * enough that this coarse-grained parallelism is beneficial despite atomic costs.
   */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) private(i)
#endif
  for (i = 0; i < n; i++){

    ReconstructionSample pt = sample[i];

    const float ptReal = pt.real;
    const float ptImag = pt.imag;
    const float ptSdc  = pt.sdc;

    if ((ptReal == 0.0f && ptImag == 0.0f) || ptSdc == 0.0f) {
      continue;
    }

    const float kx = pt.kX;
    const float ky = pt.kY;
    const float kz = pt.kZ;

    unsigned int NxL = (unsigned int)max(kx - cutoff, 0.0f);
    unsigned int NxH = (unsigned int)min(kx + cutoff, (float)(size_x - 1U));

    unsigned int NyL = (unsigned int)max(ky - cutoff, 0.0f);
    unsigned int NyH = (unsigned int)min(ky + cutoff, (float)(size_y - 1U));

    unsigned int NzL = (unsigned int)max(kz - cutoff, 0.0f);
    unsigned int NzH = (unsigned int)min(kz + cutoff, (float)(size_z - 1U));

    if (NxL > NxH || NyL > NyH || NzL > NzH) {
      continue;
    }

    const int spanX = (int)(NxH - NxL + 1U);
    const int spanY = (int)(NyH - NyL + 1U);
    const int spanZ = (int)(NzH - NzL + 1U);

    /* local small buffers to keep distance vectors in registers/cache */
    float Dx2[100];
    float Dy2[100];
    float Dz2[100];

    float *dx2 = Dx2;
    float *dy2 = Dy2;
    float *dz2 = Dz2;

    int nx, ny, nz;

    for (nz = 0; nz < spanZ; ++nz, ++dz2) {
      const float dz = kz - (float)(NzL + nz);
      *dz2 = dz * dz;
    }

    for (nx = 0; nx < spanX; ++nx, ++dx2) {
      const float dx = kx - (float)(NxL + nx);
      *dx2 = dx * dx;
    }

    for (ny = 0; ny < spanY; ++ny, ++dy2) {
      const float dy = ky - (float)(NyL + ny);
      *dy2 = dy * dy;
    }

    dz2 = Dz2;
    unsigned int idxZ = (NzL - 1U) * planeStride;

    for (nz = 0; nz < spanZ; ++nz, ++dz2) {

      idxZ += planeStride;

      if (*dz2 >= cutoff2) {
        continue;
      }

      dy2 = Dy2;
      unsigned int idxY = (NyL - 1U) * rowStride;

      for (ny = 0; ny < spanY; ++ny, ++dy2) {

        idxY += rowStride;

        const float dy2dz2 = (*dz2) + (*dy2);
        if (dy2dz2 >= cutoff2) {
          continue;
        }

        const unsigned int idx0 = idxY + idxZ;

        dx2 = Dx2;

        for (nx = 0; nx < spanX; ++nx, ++dx2) {

          const float v = dy2dz2 + (*dx2);

          if (v >= cutoff2) {
            continue;
          }

          const unsigned int idx = NxL + (unsigned int)nx + idx0;

          float w;
          if (params.useLUT) {
            w = kernel_value_LUT(v, LUT, (int)sizeLUT, invCut2) * ptSdc;
          } else {
            w = kernel_value_CPU(beta * sqrtf(1.0f - (v * invCut2))) * ptSdc;
          }

          const float contribReal = w * ptReal;
          const float contribImag = w * ptImag;

#ifdef _OPENMP
#pragma omp atomic
#endif
          gridData[idx].real += contribReal;
#ifdef _OPENMP
#pragma omp atomic
#endif
          gridData[idx].imag += contribImag;
#ifdef _OPENMP
#pragma omp atomic
#endif
          sampleDensity[idx] += 1.0f;
        }
      }
    }
  }

  return 0;
}
