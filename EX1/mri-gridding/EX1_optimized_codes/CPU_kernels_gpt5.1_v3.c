#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "UDTypes.h"

#define max(x,y) ((x<y)?y:x)
#define min(x,y) ((x>y)?y:x)

#define PI 3.14159265359

/* Mark pure computation function as static for internal linkage and
 * enable better inlining/optimization by the compiler. */
static inline float kernel_value_CPU(float v){

  const float z = v * v;

  /* Polynomials taken from
   * http://ccrma.stanford.edu/CCRMA/Courses/422/projects/kbd/kbdwindow.cpp
   * Written in Horner form; keep as-is but use const to help optimizer.
   */
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

  return -num / den;
}

void calculateLUT(float beta, float width, float** LUT, unsigned int* sizeLUT){
  if (width <= 0.0f) {
    return;
  }

  const float cutoff2 = (width * width) * 0.25f;

  /* compute size of LUT based on kernel width */
  const unsigned int size = (unsigned int)(10000.0f * width);

  /* allocate memory */
  *LUT = (float*) malloc(size * sizeof(float));

  if (*LUT == NULL) {
    *sizeLUT = 0;
    return;
  }

  const float invSize = 1.0f / (float)size;

  unsigned int k;
  for (k = 0; k < size; ++k){
    /* v in the range 0:(_width/2)^2 */
    const float v = ((float)k) * invSize * cutoff2;

    /* compute kernel value and store */
    (*LUT)[k] = kernel_value_CPU(beta * sqrtf(1.0f - (v / cutoff2)));
  }
  *sizeLUT = size;
}

static inline float kernel_value_LUT(float v, const float* __restrict LUT,
                                     int sizeLUT, float _1overCutoff2)
{
  /* v is in [0,cutoff2], map to [0,sizeLUT] */
  v *= (float)sizeLUT;
  const float scaled = v * _1overCutoff2;
  unsigned int k0 = (unsigned int)scaled;

  /* Clamp k0 to valid range to avoid out-of-bounds when accessing k0+1 */
  if (k0 >= (unsigned int)(sizeLUT - 1)) {
    k0 = (unsigned int)(sizeLUT - 2);
  }

  const float v0 = (float)k0 / _1overCutoff2;
  const float base = LUT[k0];
  const float diff = LUT[k0 + 1] - base;
  return base + ((v - v0) * diff * (1.0f / _1overCutoff2));
}

int gridding_Gold(unsigned int n,
                  parameters params,
                  ReconstructionSample* __restrict sample,
                  float* __restrict LUT,
                  unsigned int sizeLUT,
                  cmplx* __restrict gridData,
                  float* __restrict sampleDensity){

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

  const float cutoff  = ((float)(params.kernelWidth)) * 0.5f;     /* cutoff radius */
  const float cutoff2 = cutoff * cutoff;                          /* square of cutoff radius */
  const float _1overCutoff2 = 1.0f / cutoff2;                     /* 1 over square of cutoff radius */

  const float os       = params.oversample;
  const float kw       = (float)params.kernelWidth;
  const float os_mhalf = os - 0.5f;
  const float beta =
      PI * sqrtf(4.0f * kw * kw / (os * os) * (os_mhalf * os_mhalf) - 0.8f);

  const int useLUT = params.useLUT;

  unsigned int i;
  for (i = 0; i < n; i++){
    ReconstructionSample pt = sample[i];

    const float kx = pt.kX;
    const float ky = pt.kY;
    const float kz = pt.kZ;

    /* Compute bounds, clamped to grid size */
    NxL = (unsigned int)max(kx - cutoff, 0.0f);
    NxH = (unsigned int)min(kx + cutoff, (float)(size_x - 1U));

    NyL = (unsigned int)max(ky - cutoff, 0.0f);
    NyH = (unsigned int)min(ky + cutoff, (float)(size_y - 1U));

    NzL = (unsigned int)max(kz - cutoff, 0.0f);
    NzH = (unsigned int)min(kz + cutoff, (float)(size_z - 1U));

    if ((pt.real != 0.0f || pt.imag != 0.0f) && pt.sdc != 0.0f)
    {
      /* Precompute squared distances in each dimension */
      for (dz2 = Dz2, nz = (int)NzL; (unsigned int)nz <= NzH; ++nz, ++dz2)
      {
        const float dz = kz - (float)nz;
        *dz2 = dz * dz;
      }
      for (dx2 = Dx2, nx = (int)NxL; (unsigned int)nx <= NxH; ++nx, ++dx2)
      {
        const float dx = kx - (float)nx;
        *dx2 = dx * dx;
      }
      for (dy2 = Dy2, ny = (int)NyL; (unsigned int)ny <= NyH; ++ny, ++dy2)
      {
        const float dy = ky - (float)ny;
        *dy2 = dy * dy;
      }

      /* Precompute z and y strides */
      const unsigned int stride_xy = size_x * size_y;
      const unsigned int stride_x  = size_x;

      idxZ = (NzL - 1U) * stride_xy;
      for (dz2 = Dz2, nz = (int)NzL; (unsigned int)nz <= NzH; ++nz, ++dz2)
      {
        /* linear offset into 3-D matrix to get to z position */
        idxZ += stride_xy;

        idxY = (NyL - 1U) * stride_x;

        /* loop over y indexes, but only if current z distance is close enough */
        const float dz2_val = *dz2;
        if (dz2_val < cutoff2)
        {
          for (dy2 = Dy2, ny = (int)NyL; (unsigned int)ny <= NyH; ++ny, ++dy2)
          {
            idxY += stride_x;

            dy2dz2 = dz2_val + (*dy2);
            idx0   = idxY + idxZ;

            /* loop over x indexes, but only if current y+z distance is close enough */
            if (dy2dz2 < cutoff2)
            {
              for (dx2 = Dx2, nx = (int)NxL; (unsigned int)nx <= NxH; ++nx, ++dx2)
              {
                v = dy2dz2 + (*dx2);

                if (v < cutoff2)
                {
                  /* linear index of (x,y,z) point */
                  idx = (unsigned int)nx + idx0;

                  /* kernel weighting value */
                  if (useLUT){
                    w = kernel_value_LUT(v, LUT, (int)sizeLUT, _1overCutoff2) * pt.sdc;
                  } else {
                    w = kernel_value_CPU(beta * sqrtf(1.0f - (v * _1overCutoff2))) * pt.sdc;
                  }

                  /* grid data */
                  gridData[idx].real += (w * pt.real);
                  gridData[idx].imag += (w * pt.imag);

                  /* estimate sample density */
                  sampleDensity[idx] += 1.0f;
                }
              }
            }
          }
        }
      }
    }
  }

  /* Original code has no explicit return; maintain same behavior. */
  return 0;
}
