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

  for (int bx0 = 0; bx0 < blockSide; ++bx0) {
    for (int by0 = 0; by0 < blockSide; ++by0) {
      for (int bz0 = 0; bz0 < blockSide; ++bz0) {
        const int p_count = n_points[bx0][by0][bz0];
        if (p_count == 0) {
          continue;
        }
        dvector_t *const pos_p_base   = position[bx0][by0][bz0];
        dvector_t *const force_p_base = force[bx0][by0][bz0];

        const int bx1_min = MAX(0, bx0 - 1);
        const int bx1_max = MIN(blockSide, bx0 + 2);
        const int by1_min = MAX(0, by0 - 1);
        const int by1_max = MIN(blockSide, by0 + 2);
        const int bz1_min = MAX(0, bz0 - 1);
        const int bz1_max = MIN(blockSide, bz0 + 2);

        for (int bx1 = bx1_min; bx1 < bx1_max; ++bx1) {
          for (int by1 = by1_min; by1 < by1_max; ++by1) {
            for (int bz1 = bz1_min; bz1 < bz1_max; ++bz1) {
              dvector_t *const pos_q_base = position[bx1][by1][bz1];
              const int q_count = n_points[bx1][by1][bz1];
              if (q_count == 0) {
                continue;
              }

              for (int p_idx = 0; p_idx < p_count; ++p_idx) {
                const dvector_t p = pos_p_base[p_idx];
                TYPE sum_x = force_p_base[p_idx].x;
                TYPE sum_y = force_p_base[p_idx].y;
                TYPE sum_z = force_p_base[p_idx].z;

                dvector_t *qptr = pos_q_base;
                for (int q_idx = 0; q_idx < q_count; ++q_idx, ++qptr) {
                  const TYPE qx = qptr->x;
                  const TYPE qy = qptr->y;
                  const TYPE qz = qptr->z;

                  if (qx != p.x || qy != p.y || qz != p.z) {
                    const TYPE dx = p.x - qx;
                    const TYPE dy = p.y - qy;
                    const TYPE dz = p.z - qz;
                    const TYPE r2inv = 1.0 / (dx * dx + dy * dy + dz * dz);
                    const TYPE r6inv = r2inv * r2inv * r2inv;
                    const TYPE potential = r6inv * (lj1 * r6inv - lj2);
                    const TYPE f = r2inv * potential;

                    sum_x += f * dx;
                    sum_y += f * dy;
                    sum_z += f * dz;
                  }
                }

                force_p_base[p_idx].x = sum_x;
                force_p_base[p_idx].y = sum_y;
                force_p_base[p_idx].z = sum_z;
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
