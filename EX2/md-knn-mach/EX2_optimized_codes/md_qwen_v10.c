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
    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < nAtoms; i++) {
        TYPE i_x = position_x[i];
        TYPE i_y = position_y[i];
        TYPE i_z = position_z[i];
        TYPE fx = 0;
        TYPE fy = 0;
        TYPE fz = 0;

        for (int32_t j = 0; j < maxNeighbors; j++) {
            int32_t jidx = NL[i * maxNeighbors + j];
            TYPE j_x = position_x[jidx];
            TYPE j_y = position_y[jidx];
            TYPE j_z = position_z[jidx];

            TYPE delx = i_x - j_x;
            TYPE dely = i_y - j_y;
            TYPE delz = i_z - j_z;
            TYPE r2inv = 1.0 / (delx * delx + dely * dely + delz * delz);
            TYPE r6inv = r2inv * r2inv * r2inv;
            TYPE potential = r6inv * (lj1 * r6inv - lj2);
            TYPE force = r2inv * potential;

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
