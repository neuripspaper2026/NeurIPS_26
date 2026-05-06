<<<CODE>>>
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
  const int radius = (int) ceilf(cutoff * inv_gridspacing) - 1;
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
  nxcell = (int) floorf((maxext.x-minext.x) * inv_cellen) + 1;
  nycell = (int) floorf((maxext.y-minext.y) * inv_cellen) + 1;
  nzcell = (int) floorf((maxext.z-minext.z) * inv_cellen) + 1;
  ncell = nxcell * nycell * nzcell;

  /* allocate for cursor link list implementation */
  first = (int *) malloc(ncell * sizeof(int));
  for (gindex = 0;  gindex < ncell;  gindex++) {
    first[gindex] = -1;
  }
  next = (int *) malloc(natoms * sizeof(int));
  for (n = 0;  n < natoms;  n++) {
    next[n] = -1;
  }

  /* geometric hashing */
  for (n = 0;  n < natoms;  n++) {
    if (0==atom[n].q) continue;  /* skip any non-contributing atoms */
    i = (int) floorf((atom[n].x - minext.x) * inv_cellen);
    j = (int) floorf((atom[n].y - minext.y) * inv_cellen);
    k = (int) floorf((atom[n].z - minext.z) * inv_cellen);
    gindex = (k*nycell + j)*nxcell + i;
    next[n] = first[gindex];
    first[gindex] = n;
  }

  /* traverse the grid cells */
  for (gindex = 0;  gindex < ncell;  gindex++) {
    for (n = first[gindex];  n != -1;  n = next[n]) {
      x = atom[n].x - xlo;
      y = atom[n].y - ylo;
      z = atom[n].z - zlo;
      q = atom[n].q;

      /* find closest grid point with position less than or equal to atom */
      ic = (int) (x * inv_gridspacing);
      jc = (int) (y * inv_gridspacing);
      kc = (int) (z * inv_gridspacing);

      /* find extent of surrounding box of grid points */
      ia = ic - radius;
      ib = ic + radius + 1;
      ja = jc - radius;
      jb = jc + radius + 1;
      ka = kc - radius;
      kb = kc + radius + 1;

      /* trim box edges so that they are within grid point lattice */
      if (ia < 0)   ia = 0;
      if (ib >= nx) ib = nx-1;
      if (ja < 0)   ja = 0;
      if (jb >= ny) jb = ny-1;
      if (ka < 0)   ka = 0;
      if (kb >= nz) kb = nz-1;

      /* loop over surrounding grid points */
      xstart = ia*gridspacing - x;
      ystart = ja*gridspacing - y;
      dz = ka*gridspacing - z;
      for (k = ka;  k <= kb;  k++, dz += gridspacing) {
        koff = k*ny;
        dz2 = dz*dz;
        dy = ystart;
        for (j = ja;  j <= jb;  j++, dy += gridspacing) {
          jkoff = (koff + j)*nx;
          dydz2 = dy*dy + dz2;
#ifdef CHECK_CYLINDER_CPU
          if (dydz2 >= a2) continue;
#endif

          dx = xstart;
          index = jkoff + ia;
          pg = lattice->lattice + index;

#if defined(__INTEL_COMPILER)
          for (i = ia;  i <= ib;  i++, pg++, dx += gridspacing) {
            r2 = dx*dx + dydz2;
            s = (1.f - r2 * inv_a2) * (1.f - r2 * inv_a2);
            e = q * (1/sqrtf(r2)) * s;
            *pg += (r2 < a2 ? e : 0);  /* LOOP VECTORIZED!! */
          }
#else
          /* Precompute constants for inner loop */
          float dx_start = dx;
          float *pg_start = pg;
          
          /* Unroll inner loop by 4 for better performance */
          int ilimit = ib - ia + 1;
          int remainder = ilimit & 3;
          int ilim = ilimit - remainder;
          
          for (i = 0; i < ilim; i += 4) {
            float dx0 = dx_start + (i + 0) * gridspacing;
            float dx1 = dx_start + (i + 1) * gridspacing;
            float dx2 = dx_start + (i + 2) * gridspacing;
            float dx3 = dx_start + (i + 3) * gridspacing;
            
            float r2_0 = dx0*dx0 + dydz2;
            float r2_1 = dx1*dx1 + dydz2;
            float r2_2 = dx2*dx2 + dydz2;
            float r2_3 = dx3*dx3 + dydz2;
            
            if (r2_0 < a2) {
              s = (1.f - r2_0 * inv_a2);
              e = q * (1/sqrtf(r2_0)) * s * s;
              pg_start[i + 0] += e;
            }
            
            if (r2_1 < a2) {
              s = (1.f - r2_1 * inv_a2);
              e = q * (1/sqrtf(r2_1)) * s * s;
              pg_start[i + 1] += e;
            }
            
            if (r2_2 < a2) {
              s = (1.f - r2_2 * inv_a2);
              e = q * (1/sqrtf(r2_2)) * s * s;
              pg_start[i + 2] += e;
            }
            
            if (r2_3 < a2) {
              s = (1.f - r2_3 * inv_a2);
              e = q * (1/sqrtf(r2_3)) * s * s;
              pg_start[i + 3] += e;
            }
          }
          
          /* Handle remaining iterations */
          for (i = ilim; i < ilimit; i++) {
            float dx_curr = dx_start + i * gridspacing;
            r2 = dx_curr*dx_curr + dydz2;
            if (r2 >= a2) continue;
            s = (1.f - r2 *
