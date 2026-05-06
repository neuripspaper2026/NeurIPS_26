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

  const TYPE lj1_local = lj1;
  const TYPE lj2_local = lj2;

  /* Parallelize outermost block loops; use collapse to expose more work.
   * Each (b0.x,b0.y,b0.z) is independent because each thread updates a
   * disjoint region of 'force'. */
#ifdef _OPENMP
#pragma omp parallel for collapse(3) default(none) shared(n_points,force,position) firstprivate(lj1_local,lj2_local)
#endif
  for( int b0x=0; b0x<blockSide; b0x++ ) {
    for( int b0y=0; b0y<blockSide; b0y++ ) {
      for( int b0z=0; b0z<blockSide; b0z++ ) {

        const int n_p = n_points[b0x][b0y][b0z];
        /* Skip empty source blocks quickly */
        if( n_p == 0 ) {
          continue;
        }

        /* Local pointer to this block's force and position for better locality */
        dvector_t * __restrict__ pos_p_base = &position[b0x][b0y][b0z][0];
        dvector_t * __restrict__ force_p_base = &force[b0x][b0y][b0z][0];

        /* Iterate over the 3x3x3 (modulo boundary conditions) cube of blocks around b0 */
        for( int b1x = MAX(0,b0x-1); b1x < MIN(blockSide,b0x+2); b1x++ ) {
          for( int b1y = MAX(0,b0y-1); b1y < MIN(blockSide,b0y+2); b1y++ ) {
            for( int b1z = MAX(0,b0z-1); b1z < MIN(blockSide,b0z+2); b1z++ ) {

              const int q_idx_range = n_points[b1x][b1y][b1z];
              if( q_idx_range == 0 ) {
                continue;
              }

              dvector_t * __restrict__ base_q = &position[b1x][b1y][b1z][0];

              /* For all points in b0 */
              for( int p_idx=0; p_idx<n_p; p_idx++ ) {
                const dvector_t p = pos_p_base[p_idx];
                TYPE sum_x = force_p_base[p_idx].x;
                TYPE sum_y = force_p_base[p_idx].y;
                TYPE sum_z = force_p_base[p_idx].z;

                const TYPE px = p.x;
                const TYPE py = p.y;
                const TYPE pz = p.z;

                /* For all points in b1 */
                for( int q_idx=0; q_idx<q_idx_range; q_idx++ ) {
                  const dvector_t q = base_q[q_idx];

                  const TYPE qx = q.x;
                  const TYPE qy = q.y;
                  const TYPE qz = q.z;

                  /* Don't compute our own */
                  if( qx!=px || qy!=py || qz!=pz ) {
                    const TYPE dx = px - qx;
                    const TYPE dy = py - qy;
                    const TYPE dz = pz - qz;

                    const TYPE r2 = dx*dx + dy*dy + dz*dz;
                    const TYPE r2inv = 1.0 / r2;
                    const TYPE r4inv = r2inv * r2inv;
                    const TYPE r6inv = r4inv * r2inv;
                    const TYPE potential = r6inv * (lj1_local*r6inv - lj2_local);
                    const TYPE f = r2inv * potential;

                    sum_x += f*dx;
                    sum_y += f*dy;
                    sum_z += f*dz;
                  }
                } /* q loop */

                force_p_base[p_idx].x = sum_x;
                force_p_base[p_idx].y = sum_y;
                force_p_base[p_idx].z = sum_z;
              } /* p loop */

            }
          }
        } /* neighbor loops */

      }
    }
  } /* outer block loops */

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  md_grid_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                             (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
