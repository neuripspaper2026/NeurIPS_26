#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "UDTypes.h"

#define max(x,y) ((x<y)?y:x)
#define min(x,y) ((x>y)?y:x)

#define PI 3.14159265359

/* Use static const for coefficients to improve cache locality and
 * avoid re-materializing literals on each call. */
static const float kernel_num_coeffs[14] = {
  0.210580722890567e-22f,
  0.380715242345326e-19f,
  0.479440257548300e-16f,
  0.435125971262668e-13f,
  0.300931127112960e-10f,
  0.160224679395361e-7f,
  0.654858370096785e-5f,
  0.202591084143397e-2f,
  0.463076284721000e0f,
  0.754337328948189e2f,
  0.830792541809429e4f,
  0.571661130563785e6f,
  0.216415572361227e8f,
  0.356644482244025e9f
};
static const float kernel_num_last = 0.144048298227235e10f;
static const float kernel_den_c1  = -0.307646912682801e4f;
static const float kernel_den_c2  =  0.347626332405882e7f;
static const float kernel_den_c3  = -0.144048298227235e10f;

float kernel_value_CPU(float v)
{
  const float z = v * v;

  /* Horner scheme over pre-stored coefficients */
  float num = kernel_num_coeffs[0];
  num = num * z + kernel_num_coeffs[1];
  num = num * z + kernel_num_coeffs[2];
  num = num * z + kernel_num_coeffs[3];
  num = num * z + kernel_num_coeffs[4];
  num = num * z + kernel_num_coeffs[5];
  num = num * z + kernel_num_coeffs[6];
  num = num * z + kernel_num_coeffs[7];
  num = num * z + kernel_num_coeffs[8];
  num = num * z + kernel_num_coeffs[9];
  num = num * z + kernel_num_coeffs[10];
  num = num * z + kernel_num_coeffs[11];
  num = num * z + kernel_num_coeffs[12];
  num = num * z + kernel_num_coeffs[13];
  num = num * z + kernel_num_last;

  const float den = z * (z * (z + kernel_den_c1) + kernel_den_c2) + kernel_den_c3;

  return -num / den;
}

void calculateLUT(float beta, float width, float** LUT, unsigned int* sizeLUT)
{
  float v;
  const float cutoff2 = (width * width) * 0.25f;

  unsigned int size;

  if (width > 0.0f) {
    /* compute size of LUT based on kernel width */
    size = (unsigned int)(10000.0f * width);

    /* allocate memory */
    *LUT = (float*)malloc(size * sizeof(float));

    const float inv_size = 1.0f / (float)size;
    const float one_over_cutoff2 = 1.0f / cutoff2;

    unsigned int k;
    for (k = 0; k < size; ++k) {
      /* compute value to evaluate kernel at: v in the range 0:(_width/2)^2 */
      const float tk = (float)k * inv_size;
      v = tk * cutoff2;

      /* compute kernel value and store */
      /* Use fused computation to avoid repeated division where possible */
      const float ratio = 1.0f - v * one_over_cutoff2;
      (*LUT)[k] = kernel_value_CPU(beta * sqrtf(ratio));
    }
    *sizeLUT = size;
  }
}

float kernel_value_LUT(float v, float* LUT, int sizeLUT, float _1overCutoff2)
{
  /* Pre-scale once to avoid extra casts and multiplications in the hot path */
  const float scaled = v * (float)sizeLUT;
  const float idx_f = scaled * _1overCutoff2;

  /* Truncate toward zero; v is non-negative so this matches floor */
  const unsigned int k0 = (unsigned int)idx_f;
  const float v0 = (float)k0 / _1overCutoff2;

  return LUT[k0] + ((scaled - v0) * (LUT[k0 + 1] - LUT[k0]) / _1overCutoff2);
}

int gridding_Gold(unsigned int n, parameters params, ReconstructionSample* sample,
                  float* LUT, unsigned int sizeLUT, cmplx* gridData,
                  float* sampleDensity)
{
  const unsigned int size_x = (unsigned int)params.gridSize[0];
  const unsigned int size_y = (unsigned int)params.gridSize[1];
  const unsigned int size_z = (unsigned int)params.gridSize[2];

  const float cutoff  = (float)params.kernelWidth * 0.5f;      /* cutoff radius */
  const float cutoff2 = cutoff * cutoff;                       /* square of cutoff radius */
  const float _1overCutoff2 = 1.0f / cutoff2;                  /* 1 over square of cutoff radius */

  const float kw = (float)params.kernelWidth;
  const float os = params.oversample;
  const float os_m = os - 0.5f;
  const float beta =
      PI * sqrtf(4.0f * kw * kw / (os * os) * (os_m * os_m) - 0.8f);

  /* Precompute strides for 3D indexing */
  const unsigned int stride_y = size_x;
  const unsigned int stride_z = size_x * size_y;

  /* Fixed-size scratch arrays reused by each thread; bounds are ensured by kernel width
   * constraints used by this benchmark (<= 10). */
  /* Parallelize outer loop over samples: each sample accumulates contributions
   * into shared gridData/sampleDensity; writes can collide but additions are
   * commutative so we protect them with atomics. */
#pragma omp parallel
  {
    float Dx2[100];
    float Dy2[100];
    float Dz2[100];

#pragma omp for schedule(static)
    for (int i = 0; i < (int)n; i++) {
      ReconstructionSample pt = sample[i];

      const float kx = pt.kX;
      const float ky = pt.kY;
      const float kz = pt.kZ;

      /* Skip stale / empty samples early */
      if ((pt.real == 0.0f && pt.imag == 0.0f) || pt.sdc == 0.0f) {
        continue;
      }

      /* Clamp neighborhood bounds */
      const int NxL = (int)max(kx - cutoff, 0.0f);
      const int NxH = (int)min(kx + cutoff, (float)(size_x - 1U));

      const int NyL = (int)max(ky - cutoff, 0.0f);
      const int NyH = (int)min(ky + cutoff, (float)(size_y - 1U));

      const int NzL = (int)max(kz - cutoff, 0.0f);
      const int NzH = (int)min(kz + cutoff, (float)(size_z - 1U));

      int nx;
      int ny;
      int nz;

      float *dx2 = Dx2;
      float *dy2 = Dy2;
      float *dz2 = Dz2;

      /* Precompute squared distances along each axis */
      for (nz = NzL, dz2 = Dz2; nz <= NzH; ++nz, ++dz2) {
        const float dz = kz - (float)nz;
        *dz2 = dz * dz;
      }
      for (nx = NxL, dx2 = Dx2; nx <= NxH; ++nx, ++dx2) {
        const float dx = kx - (float)nx;
        *dx2 = dx * dx;
      }
      for (ny = NyL, dy2 = Dy2; ny <= NyH; ++ny, ++dy2) {
        const float dy = ky - (float)ny;
        *dy2 = dy * dy;
      }

      float pt_real = pt.real;
      float pt_imag = pt.imag;
      float pt_sdc  = pt.sdc;
      const int useLUT = params.useLUT;

      unsigned int idxZ = (unsigned int)(NzL * (int)stride_z);

      for (nz = NzL, dz2 = Dz2; nz <= NzH; ++nz, ++dz2) {
        /* linear offset into 3-D matrix to get to z position */
        if (nz > NzL) {
          idxZ += stride_z;
        }

        /* Loop over y indices, but only if current distance is close enough
         * (distance will increase by adding x&y distance) */
        if (*dz2 < cutoff2) {
          unsigned int idxY = (unsigned int)(NyL * (int)stride_y);

          for (ny = NyL, dy2 = Dy2; ny <= NyH; ++ny, ++dy2) {
            const float dy2dz2 = (*dz2) + (*dy2);

            if (ny > NyL) {
              idxY += stride_y;
            }

            if (dy2dz2 < cutoff2) {
              const unsigned int idx0 = idxY + idxZ;

              /* loop over x indexes, but only if current distance is
               * close enough (distance will increase by adding x distance) */
              const float base = dy2dz2;

              for (nx = NxL, dx2 = Dx2; nx <= NxH; ++nx, ++dx2) {
                const float v = base + (*dx2);

                if (v < cutoff2) {
                  /* linear index of (x,y,z) point */
                  const unsigned int idx = (unsigned int)nx + idx0;

                  /* kernel weighting value */
                  float w;
                  if (useLUT) {
                    w = kernel_value_LUT(v, LUT, (int)sizeLUT, _1overCutoff2) * pt_sdc;
                  } else {
                    const float ratio = 1.0f - v * _1overCutoff2;
                    w = kernel_value_CPU(beta * sqrtf(ratio)) * pt_sdc;
                  }

                  const float contrib_real = w * pt_real;
                  const float contrib_imag = w * pt_imag;

                  /* grid data: protect concurrent updates */
#pragma omp atomic
                  gridData[idx].real += contrib_real;
#pragma omp atomic
                  gridData[idx].imag += contrib_imag;

                  /* estimate sample density */
#pragma omp atomic
                  sampleDensity[idx] += 1.0f;
                }
              }
            }
          }
        }
      }
    } /* end for each sample */
  }   /* end parallel region */

  return 0;
}
