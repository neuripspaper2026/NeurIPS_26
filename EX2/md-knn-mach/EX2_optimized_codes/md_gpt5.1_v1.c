#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
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
    TYPE r6inv, potential, force, j_x, j_y, j_z;
    TYPE i_x, i_y, i_z, fx, fy, fz;

    int32_t i, j, jidx;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    const TYPE lj1_loc = lj1;
    const TYPE lj2_loc = lj2;
    const int32_t maxN_loc = maxNeighbors;
    const int32_t nAtoms_loc = nAtoms;

#ifdef _OPENMP
#pragma omp parallel for default(none) private(i, j, jidx, delx, dely, delz, r2inv, r6inv, potential, force, j_x, j_y, j_z, i_x, i_y, i_z, fx, fy, fz) shared(force_x, force_y, force_z, position_x, position_y, position_z, NL) firstprivate(lj1_loc, lj2_loc, maxN_loc, nAtoms_loc) schedule(static)
#else
    for (i = 0; i < nAtoms_loc; i++)
#endif
    {
#ifndef _OPENMP
loop_i:
#endif
        i = 
#ifdef _OPENMP
            i
#else
            i
#endif
            ;
        i_x = position_x[i];
        i_y = position_y[i];
        i_z = position_z[i];
        fx = 0.0;
        fy = 0.0;
        fz = 0.0;
#ifndef _OPENMP
loop_j:
#endif
        for (j = 0; j < maxN_loc; j++) {
            jidx = NL[i * maxN_loc + j];
            j_x = position_x[jidx];
            j_y = position_y[jidx];
            j_z = position_z[jidx];

            delx = i_x - j_x;
            dely = i_y - j_y;
            delz = i_z - j_z;

            TYPE dist2 = delx * delx + dely * dely + delz * delz;
            r2inv = 1.0 / dist2;

            r6inv = r2inv * r2inv * r2inv;

            potential = r6inv * (lj1_loc * r6inv - lj2_loc);
            force = r2inv * potential;

            fx += delx * force;
            fy += dely * force;
            fz += delz * force;
        }

        force_x[i] = fx;
        force_y[i] = fy;
        force_z[i] = fz;
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    md_knn_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                              (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
