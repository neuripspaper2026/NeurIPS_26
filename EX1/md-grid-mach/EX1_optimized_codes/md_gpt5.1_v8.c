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

        int32_t n_p = n_points[b0.x][b0.y][b0.z];
        if (n_p <= 0) {
          continue;
        }

        // Precompute neighbor block limits to avoid recomputing MIN/MAX each iteration
        int32_t b1_x_start = (b0.x > 0) ? (b0.x - 1) : 0;
        int32_t b1_x_end   = (b0.x + 2 < blockSide) ? (b0.x + 2) : blockSide;
        int32_t b1_y_start = (b0.y > 0) ? (b0.y - 1) : 0;
        int32_t b1_y_end   = (b0.y + 2 < blockSide) ? (b0.y + 2) : blockSide;
        int32_t b1_z_start = (b0.z > 0) ? (b0.z - 1) : 0;
        int32_t b1_z_end   = (b0.z + 2 < blockSide) ? (b0.z + 2) : blockSide;

        for( b1.x = b1_x_start; b1.x < b1_x_end; ++b1.x ) {
          for( b1.y = b1_y_start; b1.y < b1_y_end; ++b1.y ) {
            for( b1.z = b1_z_start; b1.z < b1_z_end; ++b1.z ) {

              dvector_t *restrict base_q = position[b1.x][b1.y][b1.z];
              int32_t q_idx_range = n_points[b1.x][b1.y][b1.z];
              if (q_idx_range <= 0) {
                continue;
              }

              dvector_t *restrict base_p_pos   = position[b0.x][b0.y][b0.z];
              dvector_t *restrict base_p_force = force[b0.x][b0.y][b0.z];

              for( p_idx = 0; p_idx < n_p; ++p_idx ) {
                p = base_p_pos[p_idx];

                TYPE sum_x = base_p_force[p_idx].x;
                TYPE sum_y = base_p_force[p_idx].y;
                TYPE sum_z = base_p_force[p_idx].z;

                for( q_idx = 0; q_idx < q_idx_range; ++q_idx ) {
                  q = base_q[q_idx];

                  // Don't compute our own
                  if( q.x == p.x && q.y == p.y && q.z == p.z ) {
                    continue;
                  }

                  dx = p.x - q.x;
                  dy = p.y - q.y;
                  dz = p.z - q.z;

                  TYPE r2 = dx*dx + dy*dy + dz*dz;
                  r2inv = 1.0 / r2;
                  r6inv = r2inv * r2inv * r2inv;
                  potential = r6inv * (lj1 * r6inv - lj2);
                  f = r2inv * potential;

                  sum_x += f * dx;
                  sum_y += f * dy;
                  sum_z += f * dz;
                }

                base_p_force[p_idx].x = sum_x;
                base_p_force[p_idx].y = sum_y;
                base_p_force[p_idx].z = sum_z;
              }
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
