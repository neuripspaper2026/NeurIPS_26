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
  const int    nx           = lattice->dim.nx;
  const int    ny           = lattice->dim.ny;
  const int    nz           = lattice->dim.nz;
  const float  xlo          = lattice->dim.lo.x;
  const float  ylo          = lattice->dim.lo.y;
  const float  zlo          = lattice->dim.lo.z;
  const float  gridspacing  = lattice->dim.h;
  const int    natoms       = atoms->size;
  Atom * const atom         = atoms->atoms;

  const float a2             = cutoff * cutoff;
  const float inv_a2         = 1.f / a2;
  const float inv_gridspacing = 1.f / gridspacing;
  const int   radius         = (int)ceilf(cutoff * inv_gridspacing) - 1;

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
  Vec3 minext, maxext;		/* Extent of atom bounding box */

  /* Start timing for total execution */
  struct timespec main_start, main_end;
  clock_gettime(CLOCK_MONOTONIC, &main_start);

  /* Start timing for kernel execution (covers entire function) */
  struct timespec kernel_start, kernel_end;
  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

#if DEBUG_PASS_RATE
  unsigned long long pass_count = 0;
  unsigned long long fail_count = 0;
#endif

  /* find min and max extent */
  get_atom_extent(&minext, &maxext, atoms);

  /* number of cells in each dimension */
  nxcell = (int) floorf((maxext.x - minext.x) * inv_cellen) + 1;
  nycell = (int) floorf((maxext.y - minext.y) * inv_cellen) + 1;
  nzcell = (int) floorf((maxext.z - minext.z) * inv_cellen) + 1;
  ncell  = nxcell * nycell * nzcell;

  /* allocate for cursor link list implementation */
  first = (int *) malloc((size_t)ncell * sizeof(int));
  next  = (int *) malloc((size_t)natoms * sizeof(int));
  if (!first || !next) {
    free(first);
    free(next);
    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    clock_gettime(CLOCK_MONOTONIC, &main_end);
    return -1;
  }

  for (gindex = 0; gindex < ncell; gindex++) {
    first[gindex] = -1;
  }
  for (n = 0; n < natoms; n++) {
    next[n] = -1;
  }

  /* geometric hashing */
  for (n = 0; n < natoms; n++) {
    const float aq = atom[n].q;
    if (aq == 0.0f) continue;  /* skip any non-contributing atoms */

    const float ax = atom[n].x;
    const float ay = atom[n].y;
    const float az = atom[n].z;

    const int ci = (int)floorf((ax - minext.x) * inv_cellen);
    const int cj = (int)floorf((ay - minext.y) * inv_cellen);
    const int ck = (int)floorf((az - minext.z) * inv_cellen);

    gindex      = (ck * nycell + cj) * nxcell + ci;
    next[n]     = first[gindex];
    first[gindex] = n;
  }

  /* precompute gridspacing-related constants */
  const float gridspacing_sq = gridspacing * gridspacing;

  /* traverse the grid cells */
  for (gindex = 0; gindex < ncell; gindex++) {
    for (n = first[gindex]; n != -1; n = next[n]) {
      x = atom[n].x - xlo;
      y = atom[n].y - ylo;
      z = atom[n].z - zlo;
      q = atom[n].q;

      /* find closest grid point with position less than or equal to atom */
      ic = (int)(x * inv_gridspacing);
      jc = (int)(y * inv_gridspacing);
      kc = (int)(z * inv_gridspacing);

      /* find extent of surrounding box of grid points */
      ia = ic - radius;
      ib = ic + radius + 1;
      ja = jc - radius;
      jb = jc + radius + 1;
      ka = kc - radius;
      kb = kc + radius + 1;

      /* trim box edges so that they are within grid point lattice */
      if (ia < 0)   ia = 0;
      if (ib >= nx) ib = nx - 1;
      if (ja < 0)   ja = 0;
      if (jb >= ny) jb = ny - 1;
      if (ka < 0)   ka = 0;
      if (kb >= nz) kb = nz - 1;

      /* loop over surrounding grid points */
      xstart = ia * gridspacing - x;
      ystart = ja * gridspacing - y;
      dz     = ka * gridspacing - z;

      for (k = ka; k <= kb; k++, dz += gridspacing) {
        koff = k * ny;
        dz2  = dz * dz;
        dy   = ystart;

        for (j = ja; j <= jb; j++, dy += gridspacing) {
          jkoff  = (koff + j) * nx;
          dydz2  = dy * dy + dz2;
#ifdef CHECK_CYLINDER_CPU
          if (dydz2 >= a2) continue;
#endif

          dx    = xstart;
          index = jkoff + ia;
          pg    = lattice->lattice + index;

#if defined(__INTEL_COMPILER)
          for (i = ia; i <= ib; i++, pg++, dx += gridspacing) {
            r2 = dx * dx + dydz2;
            s  = (1.f - r2 * inv_a2) * (1.f - r2 * inv_a2);
            e  = q * (1 / sqrtf(r2)) * s;
            *pg += (r2 < a2 ? e : 0);  /* LOOP VECTORIZED!! */
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
            const float s = (1.f - r2 * inv_a2);
            /* use r2 * inv_a2 again for slight ILP and fewer operations */
            e = q * (1.0f / sqrtf(r2)) * s * s;
            *pg += e;
          }
#endif
        }
      } /* end loop over surrounding grid points */

    } /* end loop over atoms in a gridcell */
  } /* end loop over gridcells */

  /* free memory */
  free(next);
  free(first);

#ifdef DEBUG_PASS_RATE
  printf ("Pass :%lld\n", pass_count);
  printf ("Fail :%lld\n", fail_count);
#endif

  /* End timing for kernel execution */
  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  const double kernel_time = (double)(kernel_end.tv_sec - kernel_start.tv_sec) +
                             (double)(kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;

  /* End timing for total execution */
  clock_gettime(CLOCK_MONOTONIC, &main_end);
  const double main_time = (double)(main_end.tv_sec - main_start.tv_sec) +
                           (double)(main_end.tv_nsec - main_start.tv_nsec) / 1e9;

  /* Determine timing output destination */
  FILE *timing_file = stderr;
  const char *timing_path = getenv("TIMING_LOG_FILE");
  if (timing_path && timing_path[0] != '\0') {
    FILE *tmp = fopen(timing_path, "w");
    if (tmp)
      timing_file = tmp;
  }

  /* Print timing results */
  fprintf(timing_file, "KERNEL_TIME: %.9f\n", kernel_time);
  fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);

  if (timing_file != stderr)
    fclose(timing_file);

  return 0;
}
