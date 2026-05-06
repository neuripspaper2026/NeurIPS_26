#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h> // (in path known to compiler)			needed by malloc
#include <stdio.h>  // (in path known to compiler)			needed by printf
#include <math.h>   // (in path known to compiler)			needed by exp
#ifdef _OPENMP
#include <omp.h>
#endif

#include "../lavaMD.h" // (in the main program folder)	needed to recognized input variables

#include "../util/timer/timer.h" // (in library path specified to compiler)	needed by timer

#include "kernel_cpu.h" // (in the current directory)

void kernel_cpu(par_str par, dim_str dim, box_str *box, FOUR_VECTOR *rv, fp *qv,
                FOUR_VECTOR *fv) {

    // timer
    long long time0;

    time0 = get_time();

    // timer
    long long time1;
    long long time2;
    long long time3;
    long long time4;

    // parameters
    const fp alpha = par.alpha;
    const fp a2 = (fp)(2.0f) * alpha * alpha;

    time1 = get_time();

    time2 = get_time();

    time3 = get_time();

    // Parallelize over boxes. Each box l writes only into its own fv range.
    // This avoids races because no other thread touches fv[first_i .. first_i+NUMBER_PAR_PER_BOX-1].
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long l = 0; l < dim.number_boxes; l++) {

        const long first_i = box[l].offset; // offset to common arrays

        FOUR_VECTOR * const rA = &rv[first_i];
        FOUR_VECTOR * const fA = &fv[first_i];

        const int nn = box[l].nn;

        for (int k = 0; k < (1 + nn); k++) {

            int pointer;
            if (k == 0) {
                pointer = l; // set first box to be processed to home box
            } else {
                pointer = box[l].nei[k - 1].number; // remaining boxes are neighbor boxes
            }

            const long first_j = box[pointer].offset;

            FOUR_VECTOR * const rB = &rv[first_j];
            fp * const qB = &qv[first_j];

            // cache qB for faster access and enable vectorization
            fp qB_loc[NUMBER_PAR_PER_BOX];
#pragma GCC ivdep
            for (int jj = 0; jj < NUMBER_PAR_PER_BOX; ++jj) {
                qB_loc[jj] = qB[jj];
            }

            for (int i = 0; i < NUMBER_PAR_PER_BOX; i++) {

                const fp rAi_v = rA[i].v;
                const fp rAi_x = rA[i].x;
                const fp rAi_y = rA[i].y;
                const fp rAi_z = rA[i].z;

                fp fA_v = fA[i].v;
                fp fA_x = fA[i].x;
                fp fA_y = fA[i].y;
                fp fA_z = fA[i].z;

                // explicit vectorization hint for inner loop
#pragma GCC ivdep
                for (int j = 0; j < NUMBER_PAR_PER_BOX; j++) {

                    const fp rBj_v = rB[j].v;
                    const fp rBj_x = rB[j].x;
                    const fp rBj_y = rB[j].y;
                    const fp rBj_z = rB[j].z;

                    // coefficients
                    fp r2 = rAi_v + rBj_v -
                            (rAi_x * rBj_x + rAi_y * rBj_y + rAi_z * rBj_z);
                    fp u2 = a2 * r2;
                    fp vij = expf(-u2);
                    fp fs = (fp)(2.0f) * vij;
                    fp dx = rAi_x - rBj_x;
                    fp dy = rAi_y - rBj_y;
                    fp dz = rAi_z - rBj_z;
                    fp fxij = fs * dx;
                    fp fyij = fs * dy;
                    fp fzij = fs * dz;

                    const fp qBj = qB_loc[j];

                    // forces
                    fA_v += qBj * vij;
                    fA_x += qBj * fxij;
                    fA_y += qBj * fyij;
                    fA_z += qBj * fzij;

                } // for j

                fA[i].v = fA_v;
                fA[i].x = fA_x;
                fA[i].y = fA_y;
                fA[i].z = fA_z;

            } // for i

        } // for k

    } // for l

    time4 = get_time();

    printf("Time spent in different stages of CPU/MCPU KERNEL:\n");

    printf("%15.12f s, %15.12f % : CPU/MCPU: VARIABLES\n",
           (float)(time1 - time0) / 1000000,
           (float)(time1 - time0) / (float)(time4 - time0) * 100);
    printf("%15.12f s, %15.12f % : MCPU: SET DEVICE\n",
           (float)(time2 - time1) / 1000000,
           (float)(time2 - time1) / (float)(time4 - time0) * 100);
    printf("%15.12f s, %15.12f % : CPU/MCPU: INPUTS\n",
           (float)(time3 - time2) / 1000000,
           (float)(time3 - time2) / (float)(time4 - time0) * 100);
    printf("%15.12f s, %15.12f % : CPU/MCPU: KERNEL\n",
           (float)(time4 - time3) / 1000000,
           (float)(time4 - time3) / (float)(time4 - time0) * 100);

    printf("Total time:\n");
    printf("%.12f s\n", (float)(time4 - time0) / 1000000);

} // main

#ifdef __cplusplus
}
#endif
