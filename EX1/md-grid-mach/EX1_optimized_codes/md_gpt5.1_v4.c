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
  ivector_t b0, b1;
  int32_t p_idx, q_idx;
  TYPE dx, dy, dz, r2inv, r6inv, potential, f;
  struct timespec kernel_start, kernel_end;

  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  for( b0.x = 0; b0.x < blockSide; ++b0.x ) {
    for( b0.y = 0; b0.y < blockSide; ++b0.y ) {
      for( b0.z = 0; b0.z < blockSide; ++b0.z ) {

        const int n_points_b0 = n_points[b0.x][b0.y][b0.z];

        for( b1.x = MAX(0, b0.x - 1); b1.x < MIN(blockSide, b0.x + 2); ++b1.x ) {
          for( b1.y = MAX(0, b0.y - 1); b1.y < MIN(blockSide, b0.y + 2); ++b1.y ) {
            for( b1.z = MAX(0, b0.z - 1); b1.z < MIN(blockSide, b0.z + 2); ++b1.z ) {

              dvector_t *restrict base_q = position[b1.x][b1.y][b1.z];
              const int q_idx_range = n_points[b1.x][b1.y][b1.z];

              for( p_idx = 0; p_idx < n_points_b0; ++p_idx ) {
                dvector_t p = position[b0.x][b0.y][b0.z][p_idx];

                TYPE sum_x = force[b0.x][b0.y][b0.z][p_idx].x;
                TYPE sum_y = force[b0.x][b0.y][b0.z][p_idx].y;
                TYPE sum_z = force[b0.x][b0.y][b0.z][p_idx].z;

                for( q_idx = 0; q_idx < q_idx_range; ++q_idx ) {
                  dvector_t q = base_q[q_idx];

                  if( q.x != p.x || q.y != p.y || q.z != p.z ) {
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
                }

                force[b0.x][b0.y][b0.z][p_idx].x = sum_x;
                force[b0.x][b0.y][b0.z][p_idx].y = sum_y;
                force[b0.x][b0.y][b0.z][p_idx].z = sum_z;
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
