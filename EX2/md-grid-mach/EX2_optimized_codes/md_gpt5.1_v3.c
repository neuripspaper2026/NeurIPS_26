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
  ivector_t b0, b1; // b0 is the current block, b1 is b0 or a neighboring block
  dvector_t p, q; // p is a point in b0, q is a point in either b0 or b1
  int32_t p_idx, q_idx;
  TYPE dx, dy, dz, r2inv, r6inv, potential, f;
  struct timespec kernel_start, kernel_end;

  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  const TYPE lj1_c = (TYPE)lj1;
  const TYPE lj2_c = (TYPE)lj2;

#ifdef _OPENMP
#pragma omp parallel for collapse(3) private(b0,b1,p,q,p_idx,q_idx,dx,dy,dz,r2inv,r6inv,potential,f) schedule(static)
#endif
  for( int bx0 = 0; bx0 < blockSide; bx0++ ) {
    for( int by0 = 0; by0 < blockSide; by0++ ) {
      for( int bz0 = 0; bz0 < blockSide; bz0++ ) {

        b0.x = bx0;
        b0.y = by0;
        b0.z = bz0;

        int p_count = n_points[b0.x][b0.y][b0.z];
        if (p_count <= 0)
          continue;

        dvector_t *force_base_p = &force[b0.x][b0.y][b0.z][0];
        dvector_t *pos_base_p   = &position[b0.x][b0.y][b0.z][0];

        // Iterate over the 3x3x3 (modulo boundary conditions) cube of blocks around b0
        for( b1.x = MAX(0,b0.x-1); b1.x < MIN(blockSide,b0.x+2); b1.x++ ) {
          for( b1.y = MAX(0,b0.y-1); b1.y < MIN(blockSide,b0.y+2); b1.y++ ) {
            for( b1.z = MAX(0,b0.z-1); b1.z < MIN(blockSide,b0.z+2); b1.z++ ) {

              dvector_t *base_q = &position[b1.x][b1.y][b1.z][0];
              int q_idx_range = n_points[b1.x][b1.y][b1.z];
              if (q_idx_range <= 0)
                continue;

              // For all points in b0
              for( p_idx = 0; p_idx < p_count; p_idx++ ) {
                p = pos_base_p[p_idx];
                TYPE sum_x = force_base_p[p_idx].x;
                TYPE sum_y = force_base_p[p_idx].y;
                TYPE sum_z = force_base_p[p_idx].z;

                // For all points in b1
                for( q_idx = 0; q_idx < q_idx_range; q_idx++ ) {
                  q = base_q[q_idx];

                  // Don't compute our own
                  if( (q.x != p.x) || (q.y != p.y) || (q.z != p.z) ) {
                    // Compute the LJ-potential
                    dx = p.x - q.x;
                    dy = p.y - q.y;
                    dz = p.z - q.z;
                    TYPE r2 = dx*dx + dy*dy + dz*dz;
                    r2inv = (TYPE)1.0 / r2;
                    r6inv = r2inv * r2inv * r2inv;
                    potential = r6inv * (lj1_c * r6inv - lj2_c);
                    // Update forces
                    f = r2inv * potential;
                    sum_x += f * dx;
                    sum_y += f * dy;
                    sum_z += f * dz;
                  }
                } // loop_q

                force_base_p[p_idx].x = sum_x;
                force_base_p[p_idx].y = sum_y;
                force_base_p[p_idx].z = sum_z;
              } // loop_p
            }
          }
        } // loop_grid1_*
      }
    }
  } // loop_grid0_*

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  md_grid_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                             (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
