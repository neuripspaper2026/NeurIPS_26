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
  dvector_t p, q; // p is a point in b0, q is a point in either b0 or b1
  int32_t p_idx, q_idx;
  TYPE dx, dy, dz, r2inv, r6inv, f;
  struct timespec kernel_start, kernel_end;

  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  // Iterate over the grid, block by block
  for( b0.x=0; b0.x<blockSide; b0.x++ ) {
    for( b0.y=0; b0.y<blockSide; b0.y++ ) {
      for( b0.z=0; b0.z<blockSide; b0.z++ ) {

        // Cache pointer to current block's force and position arrays
        dvector_t (*restrict pos_b0)[densityFactor] =
          position[b0.x][b0.y][b0.z];
        dvector_t (*restrict force_b0)[densityFactor] =
          force[b0.x][b0.y][b0.z];

        const int n_p = n_points[b0.x][b0.y][b0.z];

        // Iterate over the 3x3x3 (modulo boundary conditions) cube of blocks around b0
        for( b1.x=MAX(0,b0.x-1); b1.x<MIN(blockSide,b0.x+2); b1.x++ ) {
          for( b1.y=MAX(0,b0.y-1); b1.y<MIN(blockSide,b0.y+2); b1.y++ ) {
            for( b1.z=MAX(0,b0.z-1); b1.z<MIN(blockSide,b0.z+2); b1.z++ ) {

              dvector_t *restrict base_q = position[b1.x][b1.y][b1.z];
              const int q_idx_range = n_points[b1.x][b1.y][b1.z];

              // For all points in b0
              for( p_idx=0; p_idx<n_p; p_idx++ ) {
                p = pos_b0[0][p_idx];

                TYPE sum_x = force_b0[0][p_idx].x;
                TYPE sum_y = force_b0[0][p_idx].y;
                TYPE sum_z = force_b0[0][p_idx].z;

                const TYPE px = p.x;
                const TYPE py = p.y;
                const TYPE pz = p.z;

                // For all points in b1
                for( q_idx=0; q_idx<q_idx_range; q_idx++ ) {
                  q = base_q[q_idx];

                  // Don't compute our own
                  if( q.x!=px || q.y!=py || q.z!=pz ) {
                    // Compute the LJ-potential
                    dx = px - q.x;
                    dy = py - q.y;
                    dz = pz - q.z;
                    const TYPE r2 = dx*dx + dy*dy + dz*dz;
                    r2inv = 1.0 / r2;
                    r6inv = r2inv*r2inv*r2inv;
                    const TYPE potential = r6inv*(lj1*r6inv - lj2);
                    // Update forces
                    f = r2inv*potential;
                    sum_x += f*dx;
                    sum_y += f*dy;
                    sum_z += f*dz;
                  }
                } // loop_q

                force_b0[0][p_idx].x = sum_x;
                force_b0[0][p_idx].y = sum_y;
                force_b0[0][p_idx].z = sum_z;
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
