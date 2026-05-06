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
    const TYPE one = 1.0;
    const TYPE lj1_c = lj1;
    const TYPE lj2_c = lj2;

    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    for (int32_t i = 0; i < nAtoms; ++i) {
        const TYPE i_x = position_x[i];
        const TYPE i_y = position_y[i];
        const TYPE i_z = position_z[i];

        TYPE fx = 0.0;
        TYPE fy = 0.0;
        TYPE fz = 0.0;

        const int32_t base = i * maxNeighbors;

        for (int32_t j = 0; j < maxNeighbors; ++j) {
            const int32_t jidx = NL[base + j];

            const TYPE j_x = position_x[jidx];
            const TYPE j_y = position_y[jidx];
            const TYPE j_z = position_z[jidx];

            const TYPE delx = i_x - j_x;
            const TYPE dely = i_y - j_y;
            const TYPE delz = i_z - j_z;

            const TYPE r2 = delx * delx + dely * dely + delz * delz;
            const TYPE r2inv = one / r2;

            const TYPE r4inv = r2inv * r2inv;
            const TYPE r6inv = r4inv * r2inv;

            const TYPE potential = r6inv * (lj1_c * r6inv - lj2_c);
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
