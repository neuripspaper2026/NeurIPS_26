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

  // Iterate over the grid, block by block
  loop_grid0_x: for( b0.x=0; b0.x<blockSide; b0.x++ ) {
  loop_grid0_y: for( b0.y=0; b0.y<blockSide; b0.y++ ) {
  loop_grid0_z: for( b0.z=0; b0.z<blockSide; b0.z++ ) {

    int p_count = n_points[b0.x][b0.y][b0.z];

    if (p_count <= 0)
      continue;

#ifdef _OPENMP
#pragma omp parallel for private(p_idx, p, dx, dy, dz, r2inv, r6inv, potential, f, b1, q, q_idx) schedule(static)
#endif
    for (p_idx = 0; p_idx < p_count; p_idx++) {
      dvector_t p_local = position[b0.x][b0.y][b0.z][p_idx];
      TYPE sum_x = force[b0.x][b0.y][b0.z][p_idx].x;
      TYPE sum_y = force[b0.x][b0.y][b0.z][p_idx].y;
      TYPE sum_z = force[b0.x][b0.y][b0.z][p_idx].z;

      // Iterate over the 3x3x3 (modulo boundary conditions) cube of blocks around b0
      for( b1.x=MAX(0,b0.x-1); b1.x<MIN(blockSide,b0.x+2); b1.x++ ) {
        for( b1.y=MAX(0,b0.y-1); b1.y<MIN(blockSide,b0.y+2); b1.y++ ) {
          for( b1.z=MAX(0,b0.z-1); b1.z<MIN(blockSide,b0.z+2); b1.z++ ) {
            dvector_t *base_q = position[b1.x][b1.y][b1.z];
            int q_idx_range = n_points[b1.x][b1.y][b1.z];

            for( q_idx=0; q_idx<q_idx_range; q_idx++ ) {
              q = base_q[q_idx];

              // Don't compute our own
              if( q.x!=p_local.x || q.y!=p_local.y || q.z!=p_local.z ) {
                // Compute the LJ-potential
                dx = p_local.x - q.x;
                dy = p_local.y - q.y;
                dz = p_local.z - q.z;
                TYPE r2 = dx*dx + dy*dy + dz*dz;
                r2inv = 1.0 / r2;
                r6inv = r2inv * r2inv * r2inv;
                potential = r6inv * (lj1 * r6inv - lj2);
                // Update forces
                f = r2inv * potential;
                sum_x += f * dx;
                sum_y += f * dy;
                sum_z += f * dz;
              }
            } // loop_q
          } // loop_grid1_z
        } // loop_grid1_y
      } // loop_grid1_x

      force[b0.x][b0.y][b0.z][p_idx].x = sum_x;
      force[b0.x][b0.y][b0.z][p_idx].y = sum_y;
      force[b0.x][b0.y][b0.z][p_idx].z = sum_z;
    } // loop_p

  }}} // loop_grid0_*

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  md_grid_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                             (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
