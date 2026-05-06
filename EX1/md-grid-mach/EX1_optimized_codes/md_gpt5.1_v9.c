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
  struct timespec kernel_start, kernel_end;
  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  const TYPE lj1c = (TYPE)lj1;
  const TYPE lj2c = (TYPE)lj2;

  for (int bx0 = 0; bx0 < blockSide; ++bx0) {
    for (int by0 = 0; by0 < blockSide; ++by0) {
      for (int bz0 = 0; bz0 < blockSide; ++bz0) {

        const int p_range = n_points[bx0][by0][bz0];
        if (__builtin_expect(p_range == 0, 0))
          continue;

        const int bx1_min = MAX(0, bx0 - 1);
        const int bx1_max = MIN(blockSide, bx0 + 2);
        const int by1_min = MAX(0, by0 - 1);
        const int by1_max = MIN(blockSide, by0 + 2);
        const int bz1_min = MAX(0, bz0 - 1);
        const int bz1_max = MIN(blockSide, bz0 + 2);

        for (int bx1 = bx1_min; bx1 < bx1_max; ++bx1) {
          for (int by1 = by1_min; by1 < by1_max; ++by1) {
            for (int bz1 = bz1_min; bz1 < bz1_max; ++bz1) {

              dvector_t *restrict base_q = &position[bx1][by1][bz1][0];
              const int q_range = n_points[bx1][by1][bz1];
              if (__builtin_expect(q_range == 0, 0))
                continue;

              for (int p_idx = 0; p_idx < p_range; ++p_idx) {
                const dvector_t p = position[bx0][by0][bz0][p_idx];

                TYPE sum_x = force[bx0][by0][bz0][p_idx].x;
                TYPE sum_y = force[bx0][by0][bz0][p_idx].y;
                TYPE sum_z = force[bx0][by0][bz0][p_idx].z;

                const TYPE px = p.x;
                const TYPE py = p.y;
                const TYPE pz = p.z;

                for (int q_idx = 0; q_idx < q_range; ++q_idx) {
                  const dvector_t q = base_q[q_idx];

                  if (q.x == px && q.y == py && q.z == pz)
                    continue;

                  const TYPE dx = px - q.x;
                  const TYPE dy = py - q.y;
                  const TYPE dz = pz - q.z;

                  const TYPE r2 = dx*dx + dy*dy + dz*dz;
                  const TYPE r2inv = (TYPE)1.0 / r2;
                  const TYPE r6inv = r2inv * r2inv * r2inv;
                  const TYPE potential = r6inv * (lj1c * r6inv - lj2c);
                  const TYPE f = r2inv * potential;

                  sum_x += f * dx;
                  sum_y += f *dy;
                  sum_z += f * dz;
                }

                force[bx0][by0][bz0][p_idx].x = sum_x;
                force[bx0][by0][bz0][p_idx].y = sum_y;
                force[bx0][by0][bz0][p_idx].z = sum_z;
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
