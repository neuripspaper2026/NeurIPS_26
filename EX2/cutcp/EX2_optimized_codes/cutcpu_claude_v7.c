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

  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
    
    int local_gindex, local_n;
    int local_i, local_j, local_k;
    int local_ia, local_ib, local_ic;
    int local_ja, local_jb, local_jc;
    int local_ka, local_kb, local_kc;
    int local_index;
    int local_koff, local_jkoff;
    
    float local_x, local_y, local_z, local_q;
    float local_dx, local_dy, local_dz;
    float local_dz2, local_dydz2, local_r2;
    float local_e, local_s;
    float local_xstart, local_ystart;
    float *local_pg;
    
    for (local_gindex = tid; local_gindex < ncell; local_gindex += nthreads) {
      for (local_n = first[local_gindex]; local_n != -1; local_n = next[local_n]) {
        local_x = atom[local_n].x - xlo;
        local_y = atom[local_n].y - ylo;
        local_z = atom[local_n].z - zlo;
        local_q = atom[local_n].q;

        local_ic = (int) (local_x * inv_gridspacing);
        local_jc = (int) (local_y * inv_gridspacing);
        local_kc = (int) (local_z * inv_gridspacing);

        local_ia = local_ic - radius;
        local_ib = local_ic + radius + 1;
        local_ja = local_jc - radius;
        local_jb = local_jc + radius + 1;
        local_ka = local_kc - radius;
        local_kb = local_kc + radius + 1;

        if (local_ia < 0)   local_ia = 0;
        if (local_ib >= nx) local_ib = nx-1;
        if (local_ja < 0)   local_ja = 0;
        if (local_jb >= ny) local_jb = ny-1;
        if (local_ka < 0)   local_ka = 0;
        if (local_kb >= nz) local_kb = nz-1;

        local_xstart = local_ia*gridspacing - local_x;
        local_ystart = local_ja*gridspacing - local_y;
        local_dz = local_ka*gridspacing - local_z;
        
        for (local_k = local_ka; local_k <= local_kb; local_k++, local_dz += gridspacing) {
          local_koff = local_k*ny;
          local_dz2 = local_dz*local_dz;
          local_dy = local_ystart;
          
          for (local_j = local_ja; local_j <= local_jb; local_j++, local_dy += gridspacing) {
            local_jkoff = (local_koff + local_j)*nx;
            local_dydz2 = local_dy*local_dy + local_dz2;
#ifdef CHECK_CYLINDER_CPU
            if (local_dydz2 >= a2) continue;
#endif

            local_dx = local_xstart;
            local_index = local_jkoff + local_ia;
            local_pg = lattice->lattice + local_index;

            for (local_i = local_ia; local_i <= local_ib; local_i++, local_pg++, local_dx += gridspacing) {
              local_r2 = local_dx*local_dx + local_dydz2;
              if (local_r2 >= a2) {
#ifdef DEBUG_PASS_RATE
                #pragma omp atomic
                fail_count++;
#endif
                continue;
              }
#ifdef DEBUG_PASS_RATE
              #pragma omp atomic
              pass_count++;
#endif
              local_s = (1.f - local_r2 * inv_a2);
              local_e = local_q * (1.f/sqrtf(local_r2)) * local_s * local_s;
              
              #pragma omp atomic
              *local_pg += local_e;
            }
          }
        }
      }
    }
  }

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
