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

    const TYPE lj1_local = (TYPE)lj1;
    const TYPE lj2_local = (TYPE)lj2;
    const int32_t maxN = maxNeighbors;
    const int32_t nA = nAtoms;

    // Main computation
#ifdef _OPENMP
    #pragma omp parallel for default(none) shared(force_x, force_y, force_z, position_x, position_y, position_z, NL, lj1_local, lj2_local, maxN, nA) schedule(static)
#endif
    for (int32_t i = 0; i < nA; i++) {
        TYPE i_x = position_x[i];
        TYPE i_y = position_y[i];
        TYPE i_z = position_z[i];

        TYPE fx = 0.0;
        TYPE fy = 0.0;
        TYPE fz = 0.0;

        const int32_t base = i * maxN;

        for (int32_t j = 0; j < maxN; j++) {
            const int32_t jidx = NL[base + j];

            const TYPE j_x = position_x[jidx];
            const TYPE j_y = position_y[jidx];
            const TYPE j_z = position_z[jidx];

            const TYPE delx = i_x - j_x;
            const TYPE dely = i_y - j_y;
            const TYPE delz = i_z - j_z;

            const TYPE r2inv = (TYPE)1.0 / (delx * delx + dely * dely + delz * delz);
            const TYPE r6inv = r2inv * r2inv * r2inv;

            const TYPE potential = r6inv * (lj1_local * r6inv - lj2_local);
            const TYPE force = r2inv * potential;

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
