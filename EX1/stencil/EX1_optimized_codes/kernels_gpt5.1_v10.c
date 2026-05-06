#include "common.h"

void cpu_stencil(float c0, float c1, float *A0, float *Anext, const int nx, const int ny, const int nz)
{
  const int nxny = nx * ny;
  const int nxm1 = nx - 1;
  const int nym1 = ny - 1;
  const int nzm1 = nz - 1;

  for (int k = 1; k < nzm1; ++k) {
    const int k_offset = k * nxny;
    const int k_plus_offset = (k + 1) * nxny;
    const int k_minus_offset = (k - 1) * nxny;

    for (int j = 1; j < nym1; ++j) {
      const int j_offset       = j * nx;
      const int j_plus_offset  = (j + 1) * nx;
      const int j_minus_offset = (j - 1) * nx;

      const int base_idx      = k_offset      + j_offset;
      const int base_idx_kp   = k_plus_offset + j_offset;
      const int base_idx_km   = k_minus_offset + j_offset;
      const int base_idx_jp   = k_offset      + j_plus_offset;
      const int base_idx_jm   = k_offset      + j_minus_offset;

      for (int i = 1; i < nxm1; ++i) {
        const int idx    = base_idx + i;
        const int idx_ip = idx + 1;
        const int idx_im = idx - 1;

        const float center = A0[idx];

        const float neighbor_sum =
            A0[base_idx_kp + i] +
            A0[base_idx_km + i] +
            A0[base_idx_jp + i] +
            A0[base_idx_jm + i] +
            A0[idx_ip] +
            A0[idx_im];

        Anext[idx] = neighbor_sum * c1 - center * c0;
      }
    }
  }
}


