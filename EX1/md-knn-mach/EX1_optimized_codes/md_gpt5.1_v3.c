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
    const TYPE lj1_local = (TYPE)lj1;
    const TYPE lj2_local = (TYPE)lj2;
    const int32_t maxNeighbors_local = maxNeighbors;
    const int32_t nAtoms_local = nAtoms;

    TYPE delx, dely, delz;
    TYPE r2inv, r6inv, potential, force;
    TYPE j_x, j_y, j_z;
    TYPE i_x, i_y, i_z, fx, fy, fz;

    int32_t i, j, jidx;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    for (i = 0; i < nAtoms_local; ++i) {
        const int32_t base = i * maxNeighbors_local;

        i_x = position_x[i];
        i_y = position_y[i];
        i_z = position_z[i];

        fx = (TYPE)0;
        fy = (TYPE)0;
        fz = (TYPE)0;

        for (j = 0; j < maxNeighbors_local; ++j) {
            jidx = NL[base + j];

            j_x = position_x[jidx];
            j_y = position_y[jidx];
            j_z = position_z[jidx];

            delx = i_x - j_x;
            dely = i_y - j_y;
            delz = i_z - j_z;

            const TYPE r2 = delx * delx + dely * dely + delz * delz;
            r2inv = (TYPE)1.0 / r2;

            r6inv = r2inv * r2inv * r2inv;

            potential = r6inv * (lj1_local * r6inv - lj2_local);
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
