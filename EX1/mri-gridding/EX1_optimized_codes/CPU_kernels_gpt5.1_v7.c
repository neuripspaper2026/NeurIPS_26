#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "UDTypes.h"

#define max(x,y) ((x<y)?y:x)
#define min(x,y) ((x>y)?y:x)

#define PI 3.14159265359

static inline float kernel_value_CPU(const float v){

  const float z = v * v;

  /* Horner-evaluated polynomial */
  float num =
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
    z * (z * (z - 0.307646912682801e4f) +
    0.347626332405882e7f) - 0.144048298227235e10f;

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
  float *lut = (float*) malloc(size * sizeof(float));
  if (!lut) {
    *LUT = NULL;
    *sizeLUT = 0;
    return;
  }

  const float inv_size = 1.0f / (float)size;

  for (unsigned int k = 0; k < size; ++k){
    /* v in the range 0:(_width/2)^2 */
    const float v = ((float)k) * inv_size * cutoff2;

    /* compute kernel value and store */
    lut[k] = kernel_value_CPU(beta * sqrtf(1.0f - (v / cutoff2)));
  }

  *LUT = lut;
  *sizeLUT = size;
}

static inline float kernel_value_LUT(const float v_in, const float* __restrict LUT,
                                     const int sizeLUT, const float _1overCutoff2)
{
  /* v scales linearly with sizeLUT; compute index and linear interpolation */
  float v = v_in * (float)sizeLUT;
  float idx_f = v * _1overCutoff2;
  unsigned int k0 = (unsigned int)idx_f;

  if (k0 >= (unsigned int)(sizeLUT - 1)) {
    k0 = (unsigned int)(sizeLUT - 2);
  }

  const float v0 = (float)k0 / _1overCutoff2;
  const float lut0 = LUT[k0];
  const float lut1 = LUT[k0 + 1];
  return lut0 + ((v - v0) * (lut1 - lut0) * _1overCutoff2);
}

int gridding_Gold(unsigned int n, parameters params, ReconstructionSample* sample,
                  float* LUT, unsigned int sizeLUT, cmplx* gridData,
                  float* sampleDensity){

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

  const float cutoff  = ((float)(params.kernelWidth)) * 0.5f;   /* cutoff radius */
  const float cutoff2 = cutoff * cutoff;                        /* square of cutoff radius */
  const float _1overCutoff2 = 1.0f / cutoff2;                   /* 1 / cutoff^2 */

  const float kw = (float)params.kernelWidth;
  const float os = params.oversample;
  const float t1 = 4.0f * kw * kw / (os * os);
  const float os_m = os - 0.5f;
  const float t2 = os_m * os_m;
  const float beta = PI * sqrtf(t1 * t2 - 0.8f);

  const int useLUT = params.useLUT;

  for (unsigned int i = 0; i < n; ++i){
    const ReconstructionSample pt = sample[i];

    const float kx = pt.kX;
    const float ky = pt.kY;
    const float kz = pt.kZ;

    NxL = (unsigned int)max(kx - cutoff, 0.0f);
    NxH = (unsigned int)min(kx + cutoff, (float)(size_x - 1));

    NyL = (unsigned int)max(ky - cutoff, 0.0f);
    NyH = (unsigned int)min(ky + cutoff, (float)(size_y - 1));

    NzL = (unsigned int)max(kz - cutoff, 0.0f);
    NzH = (unsigned int)min(kz + cutoff, (float)(size_z - 1));

    if ((pt.real != 0.0f || pt.imag != 0.0f) && pt.sdc != 0.0f)
    {
      const float kz_loc = kz;
      const float kx_loc = kx;
      const float ky_loc = ky;

      /* precompute squared distances in each dimension */
      dz2 = Dz2;
      for (nz = (int)NzL; nz <= (int)NzH; ++nz, ++dz2) {
        const float dz = kz_loc - (float)nz;
        *dz2 = dz * dz;
      }

      dx2 = Dx2;
      for (nx = (int)NxL; nx <= (int)NxH; ++nx, ++dx2) {
        const float dx = kx_loc - (float)nx;
        *dx2 = dx * dx;
      }

      dy2 = Dy2;
      for (ny = (int)NyL; ny <= (int)NyH; ++ny, ++dy2) {
        const float dy = ky_loc - (float)ny;
        *dy2 = dy * dy;
      }

      idxZ = (NzL - 1U) * size_x * size_y;
      dz2 = Dz2;
      for (nz = (int)NzL; nz <= (int)NzH; ++nz, ++dz2)
      {
        idxZ += size_x * size_y; /* linear offset for z position */

        idxY = (NyL - 1U) * size_x;

        const float dz2_val = *dz2;
        if (dz2_val < cutoff2)
        {
          dy2 = Dy2;
          for (ny = (int)NyL; ny <= (int)NyH; ++ny, ++dy2)
          {
            idxY += size_x; /* linear offset in addition to idxZ for y */

            dy2dz2 = dz2_val + (*dy2);
            idx0 = idxY + idxZ;

            if (dy2dz2 < cutoff2)
            {
              dx2 = Dx2;
              for (nx = (int)NxL; nx <= (int)NxH; ++nx, ++dx2)
              {
                v = dy2dz2 + (*dx2);

                if (v < cutoff2)
                {
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

  return 0;
}
