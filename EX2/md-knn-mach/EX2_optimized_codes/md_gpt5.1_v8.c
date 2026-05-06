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

    const TYPE lj1_local = (TYPE)lj1;
    const TYPE lj2_local = (TYPE)lj2;

    /* Parallelize outer loop over atoms; each iteration writes to distinct i */
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    private(i, j, jidx, delx, dely, delz, r2inv, r6inv, potential, force, \
            j_x, j_y, j_z, i_x, i_y, i_z, fx, fy, fz) \
    shared(position_x, position_y, position_z, NL, force_x, force_y, force_z) \
    schedule(static)
#endif
    for (i = 0; i < nAtoms; i++) {
        i_x = position_x[i];
        i_y = position_y[i];
        i_z = position_z[i];
        fx = (TYPE)0;
        fy = (TYPE)0;
        fz = (TYPE)0;
#pragma omp simd reduction(+:fx,fy,fz) private(jidx, j_x, j_y, j_z, delx, dely, delz, r2inv, r6inv, potential, force)
        for (j = 0; j < maxNeighbors; j++) {
            /* Get neighbor */
            jidx = NL[i*maxNeighbors + j];
            /* Look up x,y,z positions */
            j_x = position_x[jidx];
            j_y = position_y[jidx];
            j_z = position_z[jidx];
            /* Calc distance */
            delx = i_x - j_x;
            dely = i_y - j_y;
            delz = i_z - j_z;
            r2inv = (TYPE)1.0 / (delx*delx + dely*dely + delz*delz);
            /* Assume no cutoff and always account for all nodes in area */
            r6inv = r2inv * r2inv * r2inv;
            potential = r6inv * (lj1_local*r6inv - lj2_local);
            /* Sum changes in force */
            force = r2inv * potential;
            fx += delx * force;
            fy += dely * force;
            fz += delz * force;
        }
        /* Update forces after all neighbors accounted for. */
        force_x[i] = fx;
        force_y[i] = fy;
        force_z[i] = fz;
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    md_knn_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                              (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
