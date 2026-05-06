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

  // Iterate over the grid, block by block
  // Parallelize over outer block space and privatize all temporaries.
  // Use collapse to increase parallel work and schedule(static) for determinism.
#pragma omp parallel for collapse(3) schedule(static) default(none) \
  shared(n_points, force, position) if(blockSide*blockSide*blockSide > 1)
  for( int bx0 = 0; bx0 < blockSide; bx0++ ) {
    for( int by0 = 0; by0 < blockSide; by0++ ) {
      for( int bz0 = 0; bz0 < blockSide; bz0++ ) {

        ivector_t b0;
        b0.x = bx0;
        b0.y = by0;
        b0.z = bz0;

        // For all neighboring blocks b1
        for( int bx1 = MAX(0,b0.x-1); bx1 < MIN(blockSide,b0.x+2); bx1++ ) {
          for( int by1 = MAX(0,b0.y-1); by1 < MIN(blockSide,b0.y+2); by1++ ) {
            for( int bz1 = MAX(0,b0.z-1); bz1 < MIN(blockSide,b0.z+2); bz1++ ) {

              dvector_t *base_q = position[bx1][by1][bz1];
              int q_idx_range   = n_points[bx1][by1][bz1];
              int p_range       = n_points[bx0][by0][bz0];

              for( int p_idx = 0; p_idx < p_range; p_idx++ ) {
                dvector_t p = position[bx0][by0][bz0][p_idx];

                // Load force components once for better locality
                TYPE sum_x = force[bx0][by0][bz0][p_idx].x;
                TYPE sum_y = force[bx0][by0][bz0][p_idx].y;
                TYPE sum_z = force[bx0][by0][bz0][p_idx].z;

                // Unrolled inner loop to expose more ILP and reduce loop overhead
                int q_idx = 0;
                int q_limit_unrolled = q_idx_range & ~3; // multiple of 4

                for( ; q_idx < q_limit_unrolled; q_idx += 4 ) {
                  // Iteration 0
                  {
                    dvector_t q = base_q[q_idx];
                    // Don't compute our own
                    if( q.x!=p.x || q.y!=p.y || q.z!=p.z ) {
                      TYPE dx = p.x - q.x;
                      TYPE dy = p.y - q.y;
                      TYPE dz = p.z - q.z;
                      TYPE r2 = dx*dx + dy*dy + dz*dz;
                      TYPE r2inv = 1.0 / r2;
                      TYPE r6inv = r2inv * r2inv * r2inv;
                      TYPE potential = r6inv * (lj1 * r6inv - lj2);
                      TYPE f = r2inv * potential;
                      sum_x += f * dx;
                      sum_y += f * dy;
                      sum_z += f * dz;
                    }
                  }

                  // Iteration 1
                  {
                    dvector_t q = base_q[q_idx+1];
                    if( q.x!=p.x || q.y!=p.y || q.z!=p.z ) {
                      TYPE dx = p.x - q.x;
                      TYPE dy = p.y - q.y;
                      TYPE dz = p.z - q.z;
                      TYPE r2 = dx*dx + dy*dy + dz*dz;
                      TYPE r2inv = 1.0 / r2;
                      TYPE r6inv = r2inv * r2inv * r2inv;
                      TYPE potential = r6inv * (lj1 * r6inv - lj2);
                      TYPE f = r2inv * potential;
                      sum_x += f * dx;
                      sum_y += f * dy;
                      sum_z += f * dz;
                    }
                  }

                  // Iteration 2
                  {
                    dvector_t q = base_q[q_idx+2];
                    if( q.x!=p.x || q.y!=p.y || q.z!=p.z ) {
                      TYPE dx = p.x - q.x;
                      TYPE dy = p.y - q.y;
                      TYPE dz = p.z - q.z;
                      TYPE r2 = dx*dx + dy*dy + dz*dz;
                      TYPE r2inv = 1.0 / r2;
                      TYPE r6inv = r2inv * r2inv * r2inv;
                      TYPE potential = r6inv * (lj1 * r6inv - lj2);
                      TYPE f = r2inv * potential;
                      sum_x += f * dx;
                      sum_y += f * dy;
                      sum_z += f * dz;
                    }
                  }

                  // Iteration 3
                  {
                    dvector_t q = base_q[q_idx+3];
                    if( q.x!=p.x || q.y!=p.y || q.z!=p.z ) {
                      TYPE dx = p.x - q.x;
                      TYPE dy = p.y - q.y;
                      TYPE dz = p.z - q.z;
                      TYPE r2 = dx*dx + dy*dy + dz*dz;
                      TYPE r2inv = 1.0 / r2;
                      TYPE r6inv = r2inv * r2inv * r2inv;
                      TYPE potential = r6inv * (lj1 * r6inv - lj2);
                      TYPE f = r2inv * potential;
                      sum_x += f * dx;
                      sum_y += f * dy;
                      sum_z += f * dz;
                    }
                  }
                }

                // Remainder loop
                for( ; q_idx < q_idx_range; q_idx++ ) {
                  dvector_t q = base_q[q_idx];
                  if( q.x!=p.x || q.y!=p.y || q.z!=p.z ) {
                    TYPE dx = p.x - q.x;
                    TYPE dy = p.y - q.y;
                    TYPE dz = p.z - q.z;
                    TYPE r2 = dx*dx + dy*dy + dz*dz;
                    TYPE r2inv = 1.0 / r2;
                    TYPE r6inv = r2inv * r2inv * r2inv;
                    TYPE potential = r6inv * (lj1 * r6inv - lj2);
                    TYPE f = r2inv * potential;
                    sum_x += f * dx;
                    sum_y += f * dy;
                    sum_z += f * dz;
                  }
                }

                force[bx0][by0][bz0][p_idx].x = sum_x;
                force[bx0][by0][bz0][p_idx].y = sum_y;
                force[bx0][by0][bz0][p_idx].z = sum_z;
              } // loop over p
            } // bz1
          } // by1
        } // bx1
      } // bz0
    } // by0
  } // bx0

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  md_grid_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                             (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
