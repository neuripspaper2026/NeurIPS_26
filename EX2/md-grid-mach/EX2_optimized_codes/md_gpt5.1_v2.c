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

  // Hoist constants
  const TYPE lj1_local = (TYPE)lj1;
  const TYPE lj2_local = (TYPE)lj2;
  const int blockSide_local = blockSide;

  // Parallelize outermost block loop; use collapse to increase work per thread
  #pragma omp parallel for collapse(3) schedule(static) if(blockSide_local*blockSide_local*blockSide_local > 1)
  for (int bx = 0; bx < blockSide_local; bx++) {
    for (int by = 0; by < blockSide_local; by++) {
      for (int bz = 0; bz < blockSide_local; bz++) {

        // Cache base pointers for this block
        dvector_t (*restrict force_block)[densityFactor] =
            force[bx][by][bz];
        dvector_t (*restrict pos_block)[densityFactor] =
            position[bx][by][bz];

        const int np_b0 = n_points[bx][by][bz];

        // Iterate over the 3x3x3 (modulo boundary conditions) cube of neighbor blocks
        const int b1x_start = MAX(0, bx - 1);
        const int b1x_end   = MIN(blockSide_local, bx + 2);
        const int b1y_start = MAX(0, by - 1);
        const int b1y_end   = MIN(blockSide_local, by + 2);
        const int b1z_start = MAX(0, bz - 1);
        const int b1z_end   = MIN(blockSide_local, bz + 2);

        for (int nbx = b1x_start; nbx < b1x_end; nbx++) {
          for (int nby = b1y_start; nby < b1y_end; nby++) {
            for (int nbz = b1z_start; nbz < b1z_end; nbz++) {

              dvector_t *restrict base_q = position[nbx][nby][nbz];
              const int q_idx_range = n_points[nbx][nby][nbz];

              // For all points in block (bx,by,bz)
              for (int p_idx = 0; p_idx < np_b0; p_idx++) {
                dvector_t p = pos_block[p_idx];
                TYPE sum_x = force_block[p_idx].x;
                TYPE sum_y = force_block[p_idx].y;
                TYPE sum_z = force_block[p_idx].z;

                // For all points in neighbor block (nbx,nby,nbz)
                for (int q_idx = 0; q_idx < q_idx_range; q_idx++) {
                  dvector_t q = base_q[q_idx];

                  // Don't compute our own
                  if (q.x != p.x || q.y != p.y || q.z != p.z) {
                    TYPE dx = p.x - q.x;
                    TYPE dy = p.y - q.y;
                    TYPE dz = p.z - q.z;
                    TYPE dist2 = dx*dx + dy*dy + dz*dz;
                    TYPE r2inv = (TYPE)1.0 / dist2;
                    TYPE r6inv = r2inv * r2inv * r2inv;
                    TYPE potential = r6inv * (lj1_local * r6inv - lj2_local);
                    TYPE f = r2inv * potential;
                    sum_x += f * dx;
                    sum_y += f * dy;
                    sum_z += f * dz;
                  }
                } // loop_q

                force_block[p_idx].x = sum_x;
                force_block[p_idx].y = sum_y;
                force_block[p_idx].z = sum_z;
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
