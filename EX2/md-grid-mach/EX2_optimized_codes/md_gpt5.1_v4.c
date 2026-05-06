#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../md.h"

static double md_grid_kernel_time_acc = 0.0;

void reset_md_grid_kernel_time(void) { md_grid_kernel_time_acc = 0.0; }
double get_md_grid_kernel_time(void) { return md_grid_kernel_time_acc; }

#define MIN(x,y) ( (x)<(y) ? (x) : (y) )
#define MAX(x,y) ( (x)>(y) ? (x) : (y) )

void md( int n_points[blockSide][blockSide][blockSide],
         dvector_t force[blockSide][blockSide][blockSide][densityFactor],
         dvector_t position[blockSide][blockSide][blockSide][densityFactor] )
{
  struct timespec kernel_start, kernel_end;
  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  const int BS = blockSide;

#ifdef _OPENMP
#pragma omp parallel
  {
#pragma omp for collapse(3) schedule(static)
#endif
    for( int bx0 = 0; bx0 < BS; ++bx0 ) {
      for( int by0 = 0; by0 < BS; ++by0 ) {
        for( int bz0 = 0; bz0 < BS; ++bz0 ) {

          const int np0 = n_points[bx0][by0][bz0];
          if (np0 == 0) continue;

          for( int bx1 = MAX(0,bx0-1); bx1 < MIN(BS,bx0+2); ++bx1 ) {
            for( int by1 = MAX(0,by0-1); by1 < MIN(BS,by0+2); ++by1 ) {
              for( int bz1 = MAX(0,bz0-1); bz1 < MIN(BS,bz0+2); ++bz1 ) {

                dvector_t *restrict base_p = position[bx0][by0][bz0];
                dvector_t *restrict base_q = position[bx1][by1][bz1];
                dvector_t *restrict base_f = force[bx0][by0][bz0];

                const int q_idx_range = n_points[bx1][by1][bz1];
                if (q_idx_range == 0) continue;

                for( int p_idx = 0; p_idx < np0; ++p_idx ) {
                  dvector_t p = base_p[p_idx];
                  TYPE sum_x = base_f[p_idx].x;
                  TYPE sum_y = base_f[p_idx].y;
                  TYPE sum_z = base_f[p_idx].z;

#pragma omp simd reduction(+:sum_x,sum_y,sum_z)
                  for( int q_idx = 0; q_idx < q_idx_range; ++q_idx ) {
                    dvector_t q = base_q[q_idx];

                    if( q.x!=p.x || q.y!=p.y || q.z!=p.z ) {
                      TYPE dx = p.x - q.x;
                      TYPE dy = p.y - q.y;
                      TYPE dz = p.z - q.z;
                      TYPE r2 = dx*dx + dy*dy + dz*dz;
                      TYPE r2inv = 1.0 / r2;
                      TYPE r6inv = r2inv*r2inv*r2inv;
                      TYPE potential = r6inv*(lj1*r6inv - lj2);
                      TYPE f = r2inv*potential;
                      sum_x += f*dx;
                      sum_y += f*dy;
                      sum_z += f*dz;
                    }
                  }

                  base_f[p_idx].x = sum_x;
                  base_f[p_idx].y = sum_y;
                  base_f[p_idx].z = sum_z;
                }
              }
            }
          }
        }
      }
    }
#ifdef _OPENMP
  }
#endif

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  md_grid_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                             (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
