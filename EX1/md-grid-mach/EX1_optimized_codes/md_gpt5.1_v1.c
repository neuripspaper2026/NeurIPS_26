#include <time.h>
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
  dvector_t p, q;   // p is a point in b0, q is a point in either b0 or b1
  int32_t p_idx, q_idx;
  TYPE dx, dy, dz, r2inv, r6inv, potential, f;
  struct timespec kernel_start, kernel_end;

  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  // Iterate over the grid, block by block
  for( b0.x = 0; b0.x < blockSide; ++b0.x ) {
    for( b0.y = 0; b0.y < blockSide; ++b0.y ) {
      for( b0.z = 0; b0.z < blockSide; ++b0.z ) {

        const int32_t n_p0 = n_points[b0.x][b0.y][b0.z];
        if (n_p0 == 0) {
          continue;
        }

        // Iterate over the 3x3x3 (modulo boundary conditions) cube of blocks around b0
        const int bx_min = MAX(0, b0.x - 1);
        const int bx_max = MIN(blockSide, b0.x + 2);
        const int by_min = MAX(0, b0.y - 1);
        const int by_max = MIN(blockSide, b0.y + 2);
        const int bz_min = MAX(0, b0.z - 1);
        const int bz_max = MIN(blockSide, b0.z + 2);

        for( b1.x = bx_min; b1.x < bx_max; ++b1.x ) {
          for( b1.y = by_min; b1.y < by_max; ++b1.y ) {
            for( b1.z = bz_min; b1.z < bz_max; ++b1.z ) {

              const int32_t q_idx_range = n_points[b1.x][b1.y][b1.z];
              if (q_idx_range == 0) {
                continue;
              }

              dvector_t * __restrict base_q =
                  &position[b1.x][b1.y][b1.z][0];

              for( p_idx = 0; p_idx < n_p0; ++p_idx ) {
                p = position[b0.x][b0.y][b0.z][p_idx];

                TYPE sum_x = force[b0.x][b0.y][b0.z][p_idx].x;
                TYPE sum_y = force[b0.x][b0.y][b0.z][p_idx].y;
                TYPE sum_z = force[b0.x][b0.y][b0.z][p_idx].z;

                const TYPE px = p.x;
                const TYPE py = p.y;
                const TYPE pz = p.z;

                for( q_idx = 0; q_idx < q_idx_range; ++q_idx ) {
                  q = base_q[q_idx];

                  // Don't compute our own
                  if( q.x != px || q.y != py || q.z != pz ) {
                    dx = px - q.x;
                    dy = py - q.y;
                    dz = pz - q.z;
                    const TYPE dist2 = dx*dx + dy*dy + dz*dz;
                    r2inv = 1.0 / dist2;
                    r6inv = r2inv * r2inv * r2inv;
                    potential = r6inv * (lj1 * r6inv - lj2);
                    f = r2inv * potential;
                    sum_x += f * dx;
                    sum_y += f * dy;
                    sum_z += f * dz;
                  }
                }

                force[b0.x][b0.y][b0.z][p_idx].x = sum_x;
                force[b0.x][b0.y][b0.z][p_idx].y = sum_y;
                force[b0.x][b0.y][b0.z][p_idx].z = sum_z;
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
