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
    TYPE r6inv, force, j_x, j_y, j_z;
    TYPE i_x, i_y, i_z, fx, fy, fz;

    int32_t i, j, jidx;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    for (i = 0; i < nAtoms; i++) {
        i_x = position_x[i];
        i_y = position_y[i];
        i_z = position_z[i];
        fx = 0.0;
        fy = 0.0;
        fz = 0.0;

        for (j = 0; j < maxNeighbors; j++) {
            jidx = NL[i * maxNeighbors + j];

            j_x = position_x[jidx];
            j_y = position_y[jidx];
            j_z = position_z[jidx];

            delx = i_x - j_x;
            dely = i_y - j_y;
            delz = i_z - j_z;

            {
                const TYPE dx2 = delx * delx;
                const TYPE dy2 = dely * dely;
                const TYPE dz2 = delz * delz;
                r2inv = 1.0 / (dx2 + dy2 + dz2);
            }

            r6inv = r2inv * r2inv * r2inv;

            {
                const TYPE t = lj1 * r6inv - lj2;
                force = r2inv * (r6inv * t);
            }

            fx += delx * force;
            fy += dely * force;
            fz += delz * force;
        }

        force_x[i] = fx;
        force_y[i] = fy;
        force_z[i] = fz;
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    md_knn_kernel_time_acc +=
        (kernel_end.tv_sec - kernel_start.tv_sec) +
        (kernel_end.tv_nsec - kernel_start.tv_nsec) * 1e-9;
}
