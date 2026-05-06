#include <time.h>
#include "../md.h"

static double md_knn_kernel_time_acc = 0.0;

void reset_md_knn_kernel_time(void) { md_knn_kernel_time_acc = 0.0; }
double get_md_knn_kernel_time(void) { return md_knn_kernel_time_acc; }

void md_kernel(TYPE force_x[nAtoms],
               TYPE force_y[nAtoms],
               TYPE force_z[nAtoms],
               TYPE position_x[nAtoms],
               TYPE position_y[nAtoms],
               TYPE position_z[nAtoms],
               int32_t NL[nAtoms*maxNeighbors])
{
    TYPE delx, dely, delz, r2inv;
    TYPE r6inv, potential, force;
    TYPE i_x, i_y, i_z, fx, fy, fz;
    TYPE pos_x_cache[nAtoms], pos_y_cache[nAtoms], pos_z_cache[nAtoms];

    int32_t i, j, jidx;
    struct timespec kernel_start, kernel_end;

    // Pre-cache positions to improve memory access pattern
    for (i = 0; i < nAtoms; i++) {
        pos_x_cache[i] = position_x[i];
        pos_y_cache[i] = position_y[i];
        pos_z_cache[i] = position_z[i];
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

loop_i : for (i = 0; i < nAtoms; i++){
             i_x = pos_x_cache[i];
             i_y = pos_y_cache[i];
             i_z = pos_z_cache[i];
             fx = 0;
             fy = 0;
             fz = 0;
loop_j : for( j = 0; j < maxNeighbors; j++){
             // Get neighbor
             jidx = NL[i*maxNeighbors + j];
             // Look up x,y,z positions from cached arrays
             delx = i_x - pos_x_cache[jidx];
             dely = i_y - pos_y_cache[jidx];
             delz = i_z - pos_z_cache[jidx];
             r2inv = 1.0/( delx*delx + dely*dely + delz*delz );
             // Assume no cutoff and aways account for all nodes in area
             r6inv = r2inv * r2inv * r2inv;
             potential = r6inv*(lj1*r6inv - lj2);
             // Sum changes in force
             force = r2inv*potential;
             fx += delx * force;
             fy += dely * force;
             fz += delz * force;
         }
         //Update forces after all neighbors accounted for.
         force_x[i] = fx;
         force_y[i] = fy;
         force_z[i] = fz;
        }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    md_knn_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                              (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
