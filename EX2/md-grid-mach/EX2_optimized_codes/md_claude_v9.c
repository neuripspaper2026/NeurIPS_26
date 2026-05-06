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

  #pragma omp parallel for collapse(3) private(b0, b1, p, q, p_idx, q_idx, dx, dy, dz, r2inv, r6inv, potential, f) schedule(dynamic, 1)
  for( b0.x=0; b0.x<blockSide; b0.x++ ) {
  for( b0.y=0; b0.y<blockSide; b0.y++ ) {
  for( b0.z=0; b0.z<blockSide; b0.z++ ) {
    const int n_p = n_points[b0.x][b0.y][b0.z];
    
    for( b1.x=MAX(0,b0.x-1); b1.x<MIN(blockSide,b0.x+2); b1.x++ ) {
    for( b1.y=MAX(0,b0.y-1); b1.y<MIN(blockSide,b0.y+2); b1.y++ ) {
    for( b1.z=MAX(0,b0.z-1); b1.z<MIN(blockSide,b0.z+2); b1.z++ ) {
      const dvector_t * __restrict__ base_q = position[b1.x][b1.y][b1.z];
      const int q_idx_range = n_points[b1.x][b1.y][b1.z];
      
      for( p_idx=0; p_idx<n_p; p_idx++ ) {
        p = position[b0.x][b0.y][b0.z][p_idx];
        TYPE sum_x = force[b0.x][b0.y][b0.z][p_idx].x;
        TYPE sum_y = force[b0.x][b0.y][b0.z][p_idx].y;
        TYPE sum_z = force[b0.x][b0.y][b0.z][p_idx].z;
        
        const TYPE px = p.x;
        const TYPE py = p.y;
        const TYPE pz = p.z;
        
        #pragma omp simd reduction(+:sum_x,sum_y,sum_z)
        for( q_idx=0; q_idx<q_idx_range; q_idx++ ) {
          q = base_q[q_idx];
          
          dx = px - q.x;
          dy = py - q.y;
          dz = pz - q.z;
          
          const TYPE not_self = (dx != 0.0 || dy != 0.0 || dz != 0.0) ? 1.0 : 0.0;
          
          r2inv = 1.0 / (dx*dx + dy*dy + dz*dz + (1.0 - not_self));
          r6inv = r2inv * r2inv * r2inv;
          potential = r6inv * (lj1*r6inv - lj2);
          f = r2inv * potential * not_self;
          
          sum_x += f * dx;
          sum_y += f * dy;
          sum_z += f * dz;
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
