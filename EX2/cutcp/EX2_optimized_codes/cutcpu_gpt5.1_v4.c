#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
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
  float inv_cellen = INV_CELLEN;
  Vec3 minext, maxext;
  float xmin, ymin, zmin;
  float xmax, ymax, zmax;

#if DEBUG_PASS_RATE
  unsigned long long pass_count = 0;
  unsigned long long fail_count = 0;
#endif

  struct timespec main_start, main_end;
  clock_gettime(CLOCK_MONOTONIC, &main_start);

  struct timespec kernel_start, kernel_end;
  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  /* find min and max extent */
  get_atom_extent(&minext, &maxext, atoms);
  xmin = minext.x;
  ymin = minext.y;
  zmin = minext.z;
  xmax = maxext.x;
  ymax = maxext.y;
  zmax = maxext.z;

  /* number of cells in each dimension */
  nxcell = (int)floorf((xmax - xmin) * inv_cellen) + 1;
  nycell = (int)floorf((ymax - ymin) * inv_cellen) + 1;
  nzcell = (int)floorf((zmax - zmin) * inv_cellen) + 1;
  ncell  = nxcell * nycell * nzcell;

  first = (int *)malloc((size_t)ncell * sizeof(int));
  next  = (int *)malloc((size_t)natoms * sizeof(int));
  if (!first || !next) {
    free(first);
    free(next);
    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    clock_gettime(CLOCK_MONOTONIC, &main_end);

    FILE *timing_file = stderr;
    const char *timing_path = getenv("TIMING_LOG_FILE");
    if (timing_path && timing_path[0] != '\0') {
      FILE *tmp = fopen(timing_path, "w");
      if (tmp)
        timing_file = tmp;
    }

    double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                         (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
    double main_time = (main_end.tv_sec - main_start.tv_sec) +
                       (main_end.tv_nsec - main_start.tv_nsec) / 1e9;

    fprintf(timing_file, "KERNEL_TIME: %.9f\n", kernel_time);
    fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);

    if (timing_file != stderr)
      fclose(timing_file);

    return -1;
  }

  /* initialize linked list heads */
  for (gindex = 0; gindex < ncell; ++gindex) {
    first[gindex] = -1;
  }
  for (n = 0; n < natoms; ++n) {
    next[n] = -1;
  }

  /* geometric hashing: build cell lists */
  for (n = 0; n < natoms; ++n) {
    if (atom[n].q == 0.0f) continue;
    i = (int)floorf((atom[n].x - xmin) * inv_cellen);
    j = (int)floorf((atom[n].y - ymin) * inv_cellen);
    k = (int)floorf((atom[n].z - zmin) * inv_cellen);
    gindex = (k * nycell + j) * nxcell + i;
    next[n] = first[gindex];
    first[gindex] = n;
  }

  {
    float *restrict lattice_base = lattice->lattice;
    const float gridspacing_local = gridspacing;
    const float a2_local = a2;
    const float inv_a2_local = inv_a2;
    const int radius_local = radius;
    const int nx_local = nx;
    const int ny_local = ny;
    const int nz_local = nz;
    const float xlo_local = xlo;
    const float ylo_local = ylo;
    const float zlo_local = zlo;

#if DEBUG_PASS_RATE
    unsigned long long pass_count_local = 0;
    unsigned long long fail_count_local = 0;
#endif

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) default(none) \
    private(gindex,n,x,y,z,q,ic,jc,kc,ia,ib,ja,jb,ka,kb,xstart,ystart,dx,dy,dz,koff,jkoff,k,j,i,index,pg,dz2,dydz2,r2,e) \
    shared(first,next,atom,ncell,lattice_base,gridspacing_local,a2_local,inv_a2_local,radius_local,nx_local,ny_local,nz_local,xlo_local,ylo_local,zlo_local) \
    reduction(+:pass_count_local,fail_count_local)
#endif
    for (gindex = 0; gindex < ncell; ++gindex) {
      for (n = first[gindex]; n != -1; n = next[n]) {
        x = atom[n].x - xlo_local;
        y = atom[n].y - ylo_local;
        z = atom[n].z - zlo_local;
        q = atom[n].q;

        ic = (int)(x / gridspacing_local);
        jc = (int)(y / gridspacing_local);
        kc = (int)(z / gridspacing_local);

        ia = ic - radius_local;
        ib = ic + radius_local + 1;
        ja = jc - radius_local;
        jb = jc + radius_local + 1;
        ka = kc - radius_local;
        kb = kc + radius_local + 1;

        if (ia < 0)      ia = 0;
        if (ib >= nx_local) ib = nx_local - 1;
        if (ja < 0)      ja = 0;
        if (jb >= ny_local) jb = ny_local - 1;
        if (ka < 0)      ka = 0;
        if (kb >= nz_local) kb = nz_local - 1;

        xstart = ia * gridspacing_local - x;
        ystart = ja * gridspacing_local - y;
        dz = ka * gridspacing_local - z;

        for (k = ka; k <= kb; ++k, dz += gridspacing_local) {
          koff = k * ny_local;
          dz2 = dz * dz;
          dy = ystart;
          for (j = ja; j <= jb; ++j, dy += gridspacing_local) {
            jkoff = (koff + j) * nx_local;
            dydz2 = dy * dy + dz2;
#ifdef CHECK_CYLINDER_CPU
            if (dydz2 >= a2_local) continue;
#endif
            dx = xstart;
            index = jkoff + ia;
            pg = lattice_base + index;

#if defined(__INTEL_COMPILER)
            for (i = ia; i <= ib; ++i, ++pg, dx += gridspacing_local) {
              r2 = dx * dx + dydz2;
              float s = (1.f - r2 * inv_a2_local);
              s *= s;
              e = q * (1.0f / sqrtf(r2)) * s;
              *pg += (r2 < a2_local ? e : 0.0f);
            }
#else
            for (i = ia; i <= ib; ++i, ++pg, dx += gridspacing_local) {
              r2 = dx * dx + dydz2;
              if (r2 >= a2_local) {
#ifdef DEBUG_PASS_RATE
                ++fail_count_local;
#endif
                continue;
              }
#ifdef DEBUG_PASS_RATE
              ++pass_count_local;
#endif
              float s = (1.f - r2 * inv_a2_local);
              e = q * (1.0f / sqrtf(r2)) * s * s;
              *pg += e;
            }
#endif
          }
        }
      }
    }

#if DEBUG_PASS_RATE
    pass_count = pass_count_local;
    fail_count = fail_count_local;
#endif
  }

  free(next);
  free(first);

#ifdef DEBUG_PASS_RATE
  printf("Pass :%lld\n", pass_count);
  printf("Fail :%lld\n", fail_count);
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
