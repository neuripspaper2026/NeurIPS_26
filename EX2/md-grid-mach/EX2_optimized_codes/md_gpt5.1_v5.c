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

  // Precompute block neighbor bounds to reduce MIN/MAX in inner loops
  int nbx_low[blockSide], nbx_high[blockSide];
  int nby_low[blockSide], nby_high[blockSide];
  int nbz_low[blockSide], nbz_high[blockSide];

  for (int bx = 0; bx < blockSide; ++bx) {
    int lx = bx - 1;
    int hx = bx + 2;
    nbx_low[bx]  = (lx > 0) ? lx : 0;
    nbx_high[bx] = (hx < blockSide) ? hx : blockSide;
  }
  for (int by = 0; by < blockSide; ++by) {
    int ly = by - 1;
    int hy = by + 2;
    nby_low[by]  = (ly > 0) ? ly : 0;
    nby_high[by] = (hy < blockSide) ? hy : blockSide;
  }
  for (int bz = 0; bz < blockSide; ++bz) {
    int lz = bz - 1;
    int hz = bz + 2;
    nbz_low[bz]  = (lz > 0) ? lz : 0;
    nbz_high[bz] = (hz < blockSide) ? hz : blockSide;
  }

  // Iterate over the grid, block by block
#ifdef _OPENMP
#pragma omp parallel for collapse(3) private(b0,b1,p,q,p_idx,q_idx,dx,dy,dz,r2inv,r6inv,potential,f) schedule(static)
#endif
  for( b0.x=0; b0.x<blockSide; b0.x++ ) {
    for( b0.y=0; b0.y<blockSide; b0.y++ ) {
      for( b0.z=0; b0.z<blockSide; b0.z++ ) {

        const int np_b0 = n_points[b0.x][b0.y][b0.z];
        if (np_b0 == 0) continue;

        const int bx_low  = nbx_low[b0.x];
        const int bx_high = nbx_high[b0.x];
        const int by_low  = nby_low[b0.y];
        const int by_high = nby_high[b0.y];
        const int bz_low  = nbz_low[b0.z];
        const int bz_high = nbz_high[b0.z];

        for( b1.x=bx_low; b1.x<bx_high; b1.x++ ) {
          for( b1.y=by_low; b1.y<by_high; b1.y++ ) {
            for( b1.z=bz_low; b1.z<bz_high; b1.z++ ) {

              dvector_t *base_q = position[b1.x][b1.y][b1.z];
              int q_idx_range = n_points[b1.x][b1.y][b1.z];
              if (q_idx_range == 0) continue;

              for( p_idx=0; p_idx<np_b0; p_idx++ ) {
                p = position[b0.x][b0.y][b0.z][p_idx];
                TYPE sum_x = force[b0.x][b0.y][b0.z][p_idx].x;
                TYPE sum_y = force[b0.x][b0.y][b0.z][p_idx].y;
                TYPE sum_z = force[b0.x][b0.y][b0.z][p_idx].z;

#pragma omp simd private(q,dx,dy,dz,r2inv,r6inv,potential,f) reduction(+:sum_x,sum_y,sum_z)
                for( q_idx=0; q_idx< q_idx_range ; q_idx++ ) {
                  q = base_q[q_idx];

                  // Don't compute our own
                  if( q.x!=p.x || q.y!=p.y || q.z!=p.z ) {
                    // Compute the LJ-potential
                    dx = p.x - q.x;
                    dy = p.y - q.y;
                    dz = p.z - q.z;
                    r2inv = 1.0/( dx*dx + dy*dy + dz*dz );
                    r6inv = r2inv*r2inv*r2inv;
                    potential = r6inv*(lj1*r6inv - lj2);
                    // Update forces
                    f = r2inv*potential;
                    sum_x += f*dx;
                    sum_y += f*dy;
                    sum_z += f*dz;
                  }
                } // loop_q
                force[b0.x][b0.y][b0.z][p_idx].x = sum_x ;
                force[b0.x][b0.y][b0.z][p_idx].y = sum_y ;
                force[b0.x][b0.y][b0.z][p_idx].z = sum_z ;
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
