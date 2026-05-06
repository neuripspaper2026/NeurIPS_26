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
  int32_t p_idx, q_idx;
  struct timespec kernel_start, kernel_end;

  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  // Iterate over the grid, block by block
  for( b0.x = 0; b0.x < blockSide; ++b0.x ) {
    for( b0.y = 0; b0.y < blockSide; ++b0.y ) {
      for( b0.z = 0; b0.z < blockSide; ++b0.z ) {

        const int32_t np_b0 = n_points[b0.x][b0.y][b0.z];

        // Skip empty source blocks early
        if (np_b0 == 0) {
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

              dvector_t * restrict const base_q = position[b1.x][b1.y][b1.z];
              const int32_t q_idx_range = n_points[b1.x][b1.y][b1.z];

              // Skip empty neighbor blocks early
              if (q_idx_range == 0) {
                continue;
              }

              // For all points in b0
              dvector_t * restrict const base_p = position[b0.x][b0.y][b0.z];
              dvector_t * restrict const base_f = force[b0.x][b0.y][b0.z];

              for( p_idx = 0; p_idx < np_b0; ++p_idx ) {
                const dvector_t p = base_p[p_idx];

                TYPE sum_x = base_f[p_idx].x;
                TYPE sum_y = base_f[p_idx].y;
                TYPE sum_z = base_f[p_idx].z;

                // For all points in b1
                for( q_idx = 0; q_idx < q_idx_range; ++q_idx ) {
                  const dvector_t q = base_q[q_idx];

                  // Don't compute our own
                  if( q.x != p.x || q.y != p.y || q.z != p.z ) {
                    const TYPE dx = p.x - q.x;
                    const TYPE dy = p.y - q.y;
                    const TYPE dz = p.z - q.z;
                    const TYPE r2 = dx*dx + dy*dy + dz*dz;
                    const TYPE r2inv = 1.0 / r2;
                    const TYPE r6inv = r2inv * r2inv * r2inv;
                    const TYPE potential = r6inv * (lj1 * r6inv - lj2);
                    const TYPE f = r2inv * potential;

                    sum_x += f * dx;
                    sum_y += f * dy;
                    sum_z += f * dz;
                  }
                } // loop_q

                base_f[p_idx].x = sum_x;
                base_f[p_idx].y = sum_y;
                base_f[p_idx].z = sum_z;
              } // loop_p
            }
          }
        }
      }
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  md_grid_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                             (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
