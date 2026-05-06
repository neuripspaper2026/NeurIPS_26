#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#if defined(_OPENMP)
#include <omp.h>
#endif
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
  int nx = lattice->dim.nx;
  int ny = lattice->dim.ny;
  int nz = lattice->dim.nz;
  float xlo = lattice->dim.lo.x;
  float ylo = lattice->dim.lo.y;
  float zlo = lattice->dim.lo.z;
  float gridspacing = lattice->dim.h;
  int natoms = atoms->size;
  Atom *atom = atoms->atoms;

  const float a2 = cutoff * cutoff;
  const float inv_a2 = 1.f / a2;
  float s;
  const float inv_gridspacing = 1.f / gridspacing;
  const int radius = (int)ceilf(cutoff * inv_gridspacing) - 1;
    /* lattice point radius about each atom */

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
  float inv_cellen = INV_CELLEN;
  Vec3 minext, maxext;		/* Extent of atom bounding box */
  float xmin, ymin, zmin;
  float xmax, ymax, zmax;

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

  /* allocate for cursor link list implementation */
  first = (int *)malloc((size_t)ncell * sizeof(int));
  next  = (int *)malloc((size_t)natoms * sizeof(int));
  if (!first || !next) {
    free(first);
    free(next);
    /* End timing for kernel execution */
    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    /* End timing for total execution */
    clock_gettime(CLOCK_MONOTONIC, &main_end);

    double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                         (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
    double main_time = (main_end.tv_sec - main_start.tv_sec) +
                       (main_end.tv_nsec - main_start.tv_nsec) / 1e9;

    FILE *timing_file = stderr;
    const char *timing_path = getenv("TIMING_LOG_FILE");
    if (timing_path && timing_path[0] != '\0') {
      FILE *tmp = fopen(timing_path, "w");
      if (tmp)
        timing_file = tmp;
    }

    fprintf(timing_file, "KERNEL_TIME: %.9f\n", kernel_time);
    fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);

    if (timing_file != stderr)
      fclose(timing_file);

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
    if (0.0f == atom[n].q) continue;  /* skip any non-contributing atoms */
    int ci = (int)floorf((atom[n].x - xmin) * inv_cellen);
    int cj = (int)floorf((atom[n].y - ymin) * inv_cellen);
    int ck = (int)floorf((atom[n].z - zmin) * inv_cellen);
    if (ci < 0) ci = 0;
    if (cj < 0) cj = 0;
    if (ck < 0) ck = 0;
    if (ci >= nxcell) ci = nxcell - 1;
    if (cj >= nycell) cj = nycell - 1;
    if (ck >= nzcell) ck = nzcell - 1;
    gindex = (ck * nycell + cj) * nxcell + ci;
    next[n] = first[gindex];
    first[gindex] = n;
  }

  /* traverse the grid cells */
#if defined(_OPENMP)
#pragma omp parallel for default(none) schedule(static) \
  private(gindex, n, x, y, z, q, ic, jc, kc, ia, ib, ja, jb, ka, kb, \
          xstart, ystart, dz, k, koff, dz2, dy, j, jkoff, dydz2, dx, \
          i, index, pg, r2, s, e) \
  reduction(+:pass_count, fail_count) \
  shared(ncell, first, next, atom, xlo, ylo, zlo, inv_gridspacing, \
         radius, nx, ny, nz, gridspacing, lattice, a2, inv_a2)
#endif
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
          pg = lattice->lattice + index;

#if defined(__INTEL_COMPILER)
          for (i = ia; i <= ib; i++, pg++, dx += gridspacing) {
            r2 = dx * dx + dydz2;
            s = (1.f - r2 * inv_a2) * (1.f - r2 * inv_a2);
            e = q * (1.0f / sqrtf(r2)) * s;
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
            s = (1.f - r2 * inv_a2);
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

  /* For debugging: print the number of times that the test passed/failed */
#ifdef DEBUG_PASS_RATE
  printf ("Pass :%lld\n", pass_count);
  printf ("Fail :%lld\n", fail_count);
#endif

  /* End timing for kernel execution */
  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                       (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;

  /* End timing for total execution */
  clock_gettime(CLOCK_MONOTONIC, &main_end);
  double main_time = (main_end.tv_sec - main_start.tv_sec) +
                     (main_end.tv_nsec - main_start.tv_nsec) / 1e9;

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
