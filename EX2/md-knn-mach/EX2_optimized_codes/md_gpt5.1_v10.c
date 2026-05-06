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
    TYPE r6inv, potential, force;
    TYPE j_x, j_y, j_z;
    TYPE i_x, i_y, i_z, fx, fy, fz;

    int32_t i, j;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    /* Parallelize outer loop over atoms.
       Each iteration writes to distinct force_[i] and uses i-local temporaries. */
#ifdef _OPENMP
#pragma omp parallel for private(i,j,delx,dely,delz,r2inv,r6inv,potential,force, \
                                 j_x,j_y,j_z,i_x,i_y,i_z,fx,fy,fz) schedule(static)
#endif
    for (i = 0; i < nAtoms; i++) {
        const int32_t base = i * maxNeighbors;

        i_x = position_x[i];
        i_y = position_y[i];
        i_z = position_z[i];

        fx = 0.0;
        fy = 0.0;
        fz = 0.0;

        for (j = 0; j < maxNeighbors; j++) {
            const int32_t jidx = NL[base + j];

            /* Load neighbor coordinates once per neighbor */
            j_x = position_x[jidx];
            j_y = position_y[jidx];
            j_z = position_z[jidx];

            /* Compute distance vector components */
            delx = i_x - j_x;
            dely = i_y - j_y;
            delz = i_z - j_z;

            /* Inverse squared distance */
            const TYPE dist2 = delx*delx + dely*dely + delz*delz;
            r2inv = 1.0 / dist2;

            /* r^-6 and potential */
            r6inv = r2inv * r2inv * r2inv;
            potential = r6inv * (lj1 * r6inv - lj2);

            /* Force scalar and accumulate vector components */
            force = r2inv * potential;
            fx += delx * force;
            fy += dely * force;
            fz += delz * force;
        }

        /* Store accumulated forces */
        force_x[i] = fx;
        force_y[i] = fy;
        force_z[i] = fz;
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    md_knn_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                              (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
