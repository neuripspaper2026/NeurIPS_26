#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "atom.h"
#include "cutoff.h"

#undef DEBUG_PASS_RATE
#define CHECK_CYLINDER_CPU

#define CELLEN      4.f
#define INV_CELLEN  (1.f/CELLEN)

extern int cpu_compute_cutoff_potential_lattice(
    Lattice *lattice,                  /* the lattice */
    float cutoff,                      /* cutoff distance */
    Atoms *atoms                       /* array of atoms */
    )
{
  const int nx = lattice->dim.nx;
  const int ny = lattice->dim.ny;
  const int nz = lattice->dim.nz;
  const float xlo = lattice->dim.lo.x;
  const float ylo = lattice->dim.lo.y;
  const float zlo = lattice->dim.lo.z;
  const float gridspacing = lattice->dim.h;
  const int natoms = atoms->size;
  Atom *const atom = atoms->atoms;
  float *const lattice_data = lattice->lattice;

  const float a2 = cutoff * cutoff;
  const float inv_a2 = 1.f / a2;
  const float inv_gridspacing = 1.f / gridspacing;
  const int radius = (int)ceilf(cutoff * inv_gridspacing) - 1;

  int n;
  int i, j, k;
  int ia, ib, ic;
  int ja, jb, jc;
  int ka, kb, kc;
  int index;
  int koff, jkoff;

  float x, y, z, q;
  float dx, dy, dz;
  float dz2, dydz2, r2;
  float e;
  float xstart, ystart;

  float *pg;

  int gindex;
  int ncell, nxcell, nycell, nzcell;
  int *first, *next;
  const float inv_cellen = INV_CELLEN;
  Vec3 minext, maxext;
  float xmin, ymin, zmin;
  float xmax, ymax, zmax;

  struct timespec main_start, main_end;
  clock_gettime(CLOCK_MONOTONIC, &main_start);

  struct timespec kernel_start, kernel_end;
  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

#if DEBUG_PASS_RATE
  unsigned long long pass_count = 0;
  unsigned long long fail_count = 0;
#endif

  get_atom_extent(&minext, &maxext, atoms);

  xmin = minext.x;
  ymin = minext.y;
  zmin = minext.z;
  xmax = maxext.x;
  ymax = maxext.y;
  zmax = maxext.z;

  nxcell = (int)floorf((xmax - xmin) * inv_cellen) + 1;
  nycell = (int)floorf((ymax - ymin) * inv_cellen) + 1;
  nzcell = (int)floorf((zmax - zmin) * inv_cellen) + 1;
  ncell = nxcell * nycell * nzcell;

  first = (int *)malloc((size_t)ncell * sizeof(int));
  if (!first) {
    return -1;
  }
  for (gindex = 0; gindex < ncell; gindex++) {
    first[gindex] = -1;
  }

  next = (int *)malloc((size_t)natoms * sizeof(int));
  if (!next) {
    free(first);
    return -1;
  }
  for (n = 0; n < natoms; n++) {
    next[n] = -1;
  }

  for (n = 0; n < natoms; n++) {
    const float qn = atom[n].q;
    if (qn == 0.0f) continue;
    const float ax = atom[n].x;
    const float ay = atom[n].y;
    const float az = atom[n].z;
    i = (int)floorf((ax - xmin) * inv_cellen);
    j = (int)floorf((ay - ymin) * inv_cellen);
    k = (int)floorf((az - zmin) * inv_cellen);
    gindex = (k * nycell + j) * nxcell + i;
    next[n] = first[gindex];
    first[gindex] = n;
  }

  for (gindex = 0; gindex < ncell; gindex++) {
    for (n = first[gindex]; n != -1; n = next[n]) {
      x = atom[n].x - xlo;
      y = atom[n].y - ylo;
      z = atom[n].z - zlo;
      q = atom[n].q;

      ic = (int)(x * inv_gridspacing);
      jc = (int)(y * inv_gridspacing);
      kc = (int)(z * inv_gridspacing);

      ia = ic - radius;
      ib = ic + radius + 1;
      ja = jc - radius;
      jb = jc + radius + 1;
      ka = kc - radius;
      kb = kc + radius + 1;

      if (ia < 0)   ia = 0;
      if (ib >= nx) ib = nx - 1;
      if (ja < 0)   ja = 0;
      if (jb >= ny) jb = ny - 1;
      if (ka < 0)   ka = 0;
      if (kb >= nz) kb = nz - 1;

      xstart = ia * gridspacing - x;
      ystart = ja * gridspacing - y;
      dz = ka * gridspacing - z;

      for (k = ka; k <= kb; k++, dz += gridspacing) {
        koff = k * ny;
        dz2 = dz * dz;
        dy = ystart;
        for (j = ja; j <= jb; j++, dy += gridspacing) {
          jkoff = (koff + j) * nx;
          dydz2 = dy * dy + dz2;
#ifdef CHECK_CYLINDER_CPU
          if (dydz2 >= a2) continue;
#endif

          dx = xstart;
          index = jkoff + ia;
          pg = lattice_data + index;

#if defined(__INTEL_COMPILER)
          for (i = ia; i <= ib; i++, pg++, dx += gridspacing) {
            r2 = dx * dx + dydz2;
            const float one_minus = 1.f - r2 * inv_a2;
            const float s_val = one_minus * one_minus;
            e = q * (1.0f / sqrtf(r2)) * s_val;
            *pg += (r2 < a2 ? e : 0.0f);
          }
#else
          for (i = ia; i <= ib; i++, pg++, dx += gridspacing) {
            r2 = dx * dx + dydz2;
            if (r2 >= a2) {
#ifdef DEBUG_PASS_RATE
              fail_count++;
#endif
              continue;
            }
#ifdef DEBUG_PASS_RATE
            pass_count++;
#endif
            const float rinv = 1.0f / sqrtf(r2);
            const float one_minus = 1.0f - r2 * inv_a2;
            e = q * rinv * one_minus * one_minus;
            *pg += e;
          }
#endif
        }
      }
    }
  }

  free(next);
  free(first);

#ifdef DEBUG_PASS_RATE
  printf("Pass :%lld\n", pass_count);
  printf("Fail :%lld\n", fail_count);
#endif

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  const double ksec = (double)(kernel_end.tv_sec - kernel_start.tv_sec);
  const double knsec = (double)(kernel_end.tv_nsec - kernel_start.tv_nsec);
  const double kernel_time = ksec + knsec / 1e9;

  clock_gettime(CLOCK_MONOTONIC, &main_end);
  const double msec = (double)(main_end.tv_sec - main_start.tv_sec);
  const double mnsec = (double)(main_end.tv_nsec - main_start.tv_nsec);
  const double main_time = msec + mnsec / 1e9;

  FILE *timing_file = stderr;
  const char *timing_path = getenv("TIMING_LOG_FILE");
  if (timing_path && timing_path[0] != '\0') {
    FILE *tmp = fopen(timing_path, "w");
    if (tmp) {
      timing_file = tmp;
    }
  }

  fprintf(timing_file, "KERNEL_TIME: %.9f\n", kernel_time);
  fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);

  if (timing_file != stderr) {
    fclose(timing_file);
  }

  return 0;
}
