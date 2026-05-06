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
    int b1_x_min = MAX(0, b0.x-1);
    int b1_x_max = MIN(blockSide, b0.x+2);
    int b1_y_min = MAX(0, b0.y-1);
    int b1_y_max = MIN(blockSide, b0.y+2);
    int b1_z_min = MAX(0, b0.z-1);
    int b1_z_max = MIN(blockSide, b0.z+2);
    
    int n_p = n_points[b0.x][b0.y][b0.z];
    
    for( b1.x=b1_x_min; b1.x<b1_x_max; b1.x++ ) {
    for( b1.y=b1_y_min; b1.y<b1_y_max; b1.y++ ) {
    for( b1.z=b1_z_min; b1.z<b1_z_max; b1.z++ ) {
      dvector_t *base_q = position[b1.x][b1.y][b1.z];
      int q_idx_range = n_points[b1.x][b1.y][b1.z];
      
      for( p_idx=0; p_idx<n_p; p_idx++ ) {
        p = position[b0.x][b0.y][b0.z][p_idx];
        TYPE sum_x = force[b0.x][b0.y][b0.z][p_idx].x;
        TYPE sum_y = force[b0.x][b0.y][b0.z][p_idx].y;
        TYPE sum_z = force[b0.x][b0.y][b0.z][p_idx].z;
        
        TYPE px = p.x;
        TYPE py = p.y;
        TYPE pz = p.z;
        
        for( q_idx=0; q_idx<q_idx_range; q_idx++ ) {
          q = base_q[q_idx];
          
          TYPE qx = q.x;
          TYPE qy = q.y;
          TYPE qz = q.z;
          
          int same_point = (qx == px) & (qy == py) & (qz == pz);
          
          dx = px - qx;
          dy = py - qy;
          dz = pz - qz;
          
          TYPE dx2 = dx*dx;
          TYPE dy2 = dy*dy;
          TYPE dz2 = dz*dz;
          TYPE r2 = dx2 + dy2 + dz2;
          
          r2inv = 1.0 / r2;
          TYPE r4inv = r2inv * r2inv;
          r6inv = r4inv * r2inv;
          TYPE r12inv = r6inv * r6inv;
          
          potential = r12inv * lj1 - r6inv * lj2;
          f = r2inv * potential;
          
          TYPE fx = f * dx;
          TYPE fy = f * dy;
          TYPE fz = f * dz;
          
          sum_x += same_point ? 0.0 : fx;
          sum_y += same_point ? 0.0 : fy;
          sum_z += same_point ? 0.0 : fz;
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
