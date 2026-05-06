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
  ivector_t b0, b1;
  dvector_t p, q;
  int32_t p_idx, q_idx;
  TYPE dx, dy, dz, r2inv, r6inv, potential, f;
  struct timespec kernel_start, kernel_end;

  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  // Iterate over the grid, block by block
  #pragma omp parallel for collapse(3) private(b0, b1, p, q, p_idx, q_idx, dx, dy, dz, r2inv, r6inv, potential, f) schedule(dynamic, 1)
  for( b0.x=0; b0.x<blockSide; b0.x++ ) {
  for( b0.y=0; b0.y<blockSide; b0.y++ ) {
  for( b0.z=0; b0.z<blockSide; b0.z++ ) {
    // Iterate over the 3x3x3 (modulo boundary conditions) cube of blocks around b0
    for( b1.x=MAX(0,b0.x-1); b1.x<MIN(blockSide,b0.x+2); b1.x++ ) {
    for( b1.y=MAX(0,b0.y-1); b1.y<MIN(blockSide,b0.y+2); b1.y++ ) {
    for( b1.z=MAX(0,b0.z-1); b1.z<MIN(blockSide,b0.z+2); b1.z++ ) {
      // For all points in b0
      dvector_t *base_q = position[b1.x][b1.y][b1.z];
      int q_idx_range = n_points[b1.x][b1.y][b1.z];
      int p_idx_range = n_points[b0.x][b0.y][b0.z];
      
      for( p_idx=0; p_idx<p_idx_range; p_idx++ ) {
        p = position[b0.x][b0.y][b0.z][p_idx];
        TYPE sum_x = force[b0.x][b0.y][b0.z][p_idx].x;
        TYPE sum_y = force[b0.x][b0.y][b0.z][p_idx].y;
        TYPE sum_z = force[b0.x][b0.y][b0.z][p_idx].z;
        
        TYPE px = p.x;
        TYPE py = p.y;
        TYPE pz = p.z;
        
        // For all points in b1
        #pragma omp simd reduction(+:sum_x,sum_y,sum_z)
        for( q_idx=0; q_idx<q_idx_range; q_idx++ ) {
          q = base_q[q_idx];

          // Compute the LJ-potential
          dx = px - q.x;
          dy = py - q.y;
          dz = pz - q.z;
          
          // Don't compute our own (branchless version)
          int is_same = (q.x==px) & (q.y==py) & (q.z==pz);
          
          r2inv = 1.0/( dx*dx + dy*dy + dz*dz );
          r6inv = r2inv*r2inv*r2inv;
          potential = r6inv*(lj1*r6inv - lj2);
          f = r2inv*potential;
          
          // Conditional update using mask
          TYPE mask = is_same ? 0.0 : 1.0;
          sum_x += mask * f * dx;
          sum_y += mask * f * dy;
          sum_z += mask * f * dz;
        }
        
        force[b0.x][b0.y][b0.z][p_idx].x = sum_x;
        force[b0.x][b0.y][b0.z][p_idx].y = sum_y;
        force[b0.x][b0.y][b0.z][p_idx].z = sum_z;
      }
    }}}
  }}}

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  md_grid_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                             (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
