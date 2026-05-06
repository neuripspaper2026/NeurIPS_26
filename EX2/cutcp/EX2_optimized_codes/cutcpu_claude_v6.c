#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "atom.h"
#include "cutoff.h"

#undef DEBUG_PASS_RATE
#define CHECK_CYLINDER_CPU

#define CELLEN      4.f
#define INV_CELLEN  (1.f/CELLEN)

extern int cpu_compute_cutoff_potential_lattice(
    Lattice *lattice,
    float cutoff,
    Atoms *atoms
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
  const float inv_gridspacing = 1.f / gridspacing;
  const int radius = (int) ceilf(cutoff * inv_gridspacing) - 1;

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
  float e, s;
  float xstart, ystart;

  float *pg;

  int gindex;
  int ncell, nxcell, nycell, nzcell;
  int *first, *next;
  float inv_cellen = INV_CELLEN;
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

  nxcell = (int) floorf((maxext.x-minext.x) * inv_cellen) + 1;
  nycell = (int) floorf((maxext.y-minext.y) * inv_cellen) + 1;
  nzcell = (int) floorf((maxext.z-minext.z) * inv_cellen) + 1;
  ncell = nxcell * nycell * nzcell;

  first = (int *) malloc(ncell * sizeof(int));
  for (gindex = 0;  gindex < ncell;  gindex++) {
    first[gindex] = -1;
  }
  next = (int *) malloc(natoms * sizeof(int));
  for (n = 0;  n < natoms;  n++) {
    next[n] = -1;
  }

  for (n = 0;  n < natoms;  n++) {
    if (0==atom[n].q) continue;
    i = (int) floorf((atom[n].x - minext.x) * inv_cellen);
    j = (int) floorf((atom[n].y - minext.y) * inv_cellen);
    k = (int) floorf((atom[n].z - minext.z) * inv_cellen);
    gindex = (k*nycell + j)*nxcell + i;
    next[n] = first[gindex];
    first[gindex] = n;
  }

  const int lattice_size = nx * ny * nz;

#ifdef _OPENMP
  #pragma omp parallel
  {
    float *local_lattice = (float *) calloc(lattice_size, sizeof(float));
    
    #pragma omp for schedule(dynamic, 16) nowait
    for (gindex = 0;  gindex < ncell;  gindex++) {
      for (n = first[gindex];  n != -1;  n = next[n]) {
        float x = atom[n].x - xlo;
        float y = atom[n].y - ylo;
        float z = atom[n].z - zlo;
        float q = atom[n].q;

        int ic = (int) (x * inv_gridspacing);
        int jc = (int) (y * inv_gridspacing);
        int kc = (int) (z * inv_gridspacing);

        int ia = ic - radius;
        int ib = ic + radius + 1;
        int ja = jc - radius;
        int jb = jc + radius + 1;
        int ka = kc - radius;
        int kb = kc + radius + 1;

        if (ia < 0)   ia = 0;
        if (ib >= nx) ib = nx-1;
        if (ja < 0)   ja = 0;
        if (jb >= ny) jb = ny-1;
        if (ka < 0)   ka = 0;
        if (kb >= nz) kb = nz-1;

        float xstart = ia*gridspacing - x;
        float ystart = ja*gridspacing - y;
        float dz = ka*gridspacing - z;
        
        for (int k = ka;  k <= kb;  k++, dz += gridspacing) {
          int koff = k*ny;
          float dz2 = dz*dz;
          float dy = ystart;
          
          for (int j = ja;  j <= jb;  j++, dy += gridspacing) {
            int jkoff = (koff + j)*nx;
            float dydz2 = dy*dy + dz2;
#ifdef CHECK_CYLINDER_CPU
            if (dydz2 >= a2) continue;
#endif
            float dx = xstart;
            int index = jkoff + ia;
            float *pg = local_lattice + index;

            for (int i = ia;  i <= ib;  i++, pg++, dx += gridspacing) {
              float r2 = dx*dx + dydz2;
              if (r2 >= a2) continue;
              
              float s = (1.f - r2 * inv_a2);
              float e = q * (1.f/sqrtf(r2)) * s * s;
              *pg += e;
            }
          }
        }
      }
    }
    
    #pragma omp critical
    {
      for (int idx = 0; idx < lattice_size; idx++) {
        lattice->lattice[idx] += local_lattice[idx];
      }
    }
    
    free(local_lattice);
  }
#else
  for (gindex = 0;  gindex < ncell;  gindex++) {
    for (n = first[gindex];  n != -1;  n = next[n]) {
      x = atom[n].x - xlo;
      y = atom[n].y - ylo;
      z = atom[n].z - zlo;
      q = atom[n].q;

      ic = (int) (x * inv_gridspacing);
      jc = (int) (y * inv_gridspacing);
      kc = (int) (z * inv_gridspacing);

      ia = ic - radius;
      ib = ic + radius + 1;
      ja = jc - radius;
      jb = jc + radius + 1;
      ka = kc - radius;
      kb = kc + radius + 1;

      if (ia < 0)   ia = 0;
      if (ib >= nx) ib = nx-1;
      if (ja < 0)   ja = 0;
      if (jb >= ny) jb = ny-1;
      if (ka < 0)   ka = 0;
      if (kb >= nz) kb = nz-1;

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

          for (i = ia;  i <= ib;  i++, pg++, dx += gridspacing) {
            r2 = dx*dx + dydz2;
            if (r2 >= a2) continue;
            
            s = (1.f - r2 * inv_a2);
            e = q * (1/sqrtf(r2)) * s * s;
            *pg += e;
          }
        }
      }
    }
  }
#endif

  free(next);
  free(first);

#ifdef DEBUG_PASS_RATE
  printf ("Pass :%lld\n", pass_count);
  printf ("Fail :%lld\n", fail_count);
#endif

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                       (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;

  clock_gettime(CLOCK_MONOTONIC, &main_end);
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

  return 0;
}
