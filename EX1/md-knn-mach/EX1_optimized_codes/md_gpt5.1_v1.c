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
    TYPE delx, dely, delz;
    TYPE r2inv, r6inv, potential, force;
    TYPE j_x, j_y, j_z;
    TYPE i_x, i_y, i_z, fx, fy, fz;

    int32_t i, j;
    const int32_t total_neighbors = nAtoms * maxNeighbors;
    TYPE r2;

    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    /* Preload positions into local arrays to improve cache locality and allow
       the compiler to apply more aggressive optimizations. */
    TYPE pos_x_local[nAtoms];
    TYPE pos_y_local[nAtoms];
    TYPE pos_z_local[nAtoms];

    for (i = 0; i < nAtoms; ++i) {
        pos_x_local[i] = position_x[i];
        pos_y_local[i] = position_y[i];
        pos_z_local[i] = position_z[i];
    }

    /* Main force computation loop */
    for (i = 0; i < nAtoms; ++i) {
        i_x = pos_x_local[i];
        i_y = pos_y_local[i];
        i_z = pos_z_local[i];

        fx = 0.0;
        fy = 0.0;
        fz = 0.0;

        const int32_t base_idx = i * maxNeighbors;

        /* Unroll the inner neighbor loop manually to reduce loop overhead
           and expose more ILP for the compiler. */
        for (j = 0; j < maxNeighbors; j += 4) {
            int32_t jidx0 = NL[base_idx + j];
            int32_t jidx1 = NL[base_idx + j + 1];
            int32_t jidx2 = NL[base_idx + j + 2];
            int32_t jidx3 = NL[base_idx + j + 3];

            /* Neighbor 0 */
            j_x = pos_x_local[jidx0];
            j_y = pos_y_local[jidx0];
            j_z = pos_z_local[jidx0];

            delx = i_x - j_x;
            dely = i_y - j_y;
            delz = i_z - j_z;
            r2   = delx * delx + dely * dely + delz * delz;
            r2inv = 1.0 / r2;
            r6inv = r2inv * r2inv * r2inv;
            potential = r6inv * (lj1 * r6inv - lj2);
            force = r2inv * potential;
            fx += delx * force;
            fy += dely * force;
            fz += delz * force;

            /* Neighbor 1 */
            j_x = pos_x_local[jidx1];
            j_y = pos_y_local[jidx1];
            j_z = pos_z_local[jidx1];

            delx = i_x - j_x;
            dely = i_y - j_y;
            delz = i_z - j_z;
            r2   = delx * delx + dely * dely + delz * delz;
            r2inv = 1.0 / r2;
            r6inv = r2inv * r2inv * r2inv;
            potential = r6inv * (lj1 * r6inv - lj2);
            force = r2inv * potential;
            fx += delx * force;
            fy += dely * force;
            fz += delz * force;

            /* Neighbor 2 */
            j_x = pos_x_local[jidx2];
            j_y = pos_y_local[jidx2];
            j_z = pos_z_local[jidx2];

            delx = i_x - j_x;
            dely = i_y - j_y;
            delz = i_z - j_z;
            r2   = delx * delx + dely * dely + delz * delz;
            r2inv = 1.0 / r2;
            r6inv = r2inv * r2inv * r2inv;
            potential = r6inv * (lj1 * r6inv - lj2);
            force = r2inv * potential;
            fx += delx * force;
            fy += dely * force;
            fz += delz * force;

            /* Neighbor 3 */
            j_x = pos_x_local[jidx3];
            j_y = pos_y_local[jidx3];
            j_z = pos_z_local[jidx3];

            delx = i_x - j_x;
            dely = i_y - j_y;
            delz = i_z - j_z;
            r2   = delx * delx + dely * dely + delz * delz;
            r2inv = 1.0 / r2;
            r6inv = r2inv * r2inv * r2inv;
            potential = r6inv * (lj1 * r6inv - lj2);
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
