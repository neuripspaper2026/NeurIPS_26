#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h> // needed by malloc
#include <stdio.h>  // needed by printf
#include <math.h>   // needed by exp
#ifdef _OPENMP
#include <omp.h>
#endif

#include "../lavaMD.h" // needed to recognize input variables

#include "../util/timer/timer.h" // needed by timer

#include "kernel_cpu.h" // in the current directory

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
    const fp a2 = (fp)2.0 * alpha * alpha;

    time1 = get_time();

    time2 = get_time();

    time3 = get_time();

    // main computation
    // outer loop over boxes: parallelized
#ifdef _OPENMP
#pragma omp parallel
    {
#endif

        // thread-private temporaries
        long l;
        int k, i, j;
        long first_i, first_j;
        int pointer;
        FOUR_VECTOR *rA;
        FOUR_VECTOR *fA;
        FOUR_VECTOR *rB;
        fp *qB;

        fp r2;
        fp u2;
        fp vij;
        fp fs;
        fp fxij, fyij, fzij;
        THREE_VECTOR d;

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (l = 0; l < dim.number_boxes; l++) {

            first_i = box[l].offset; // offset to common arrays

            rA = &rv[first_i];
            fA = &fv[first_i];

            // loop over home + neighbor boxes
            for (k = 0; k < (1 + box[l].nn); k++) {

                if (k == 0) {
                    pointer = l; // set first box to be processed to home box
                } else {
                    pointer = box[l].nei[k - 1].number; // neighbor boxes
                }

                first_j = box[pointer].offset;

                rB = &rv[first_j];
                qB = &qv[first_j];

                // particles in home box
                for (i = 0; i < NUMBER_PAR_PER_BOX; i++) {

                    // accumulate contributions to particle i in registers
                    fp fij_v = fA[i].v;
                    fp fij_x = fA[i].x;
                    fp fij_y = fA[i].y;
                    fp fij_z = fA[i].z;

#pragma omp simd aligned(rA, rB, qB : 16) reduction(+ : fij_v, fij_x, fij_y, fij_z)
                    for (j = 0; j < NUMBER_PAR_PER_BOX; j++) {

                        // coefficients
                        r2 = rA[i].v + rB[j].v -
                             (rA[i].x * rB[j].x + rA[i].y * rB[j].y +
                              rA[i].z * rB[j].z);
                        u2 = a2 * r2;
                        vij = expf(-u2);
                        fs = (fp)2.0 * vij;

                        d.x = rA[i].x - rB[j].x;
                        d.y = rA[i].y - rB[j].y;
                        d.z = rA[i].z - rB[j].z;

                        fxij = fs * d.x;
                        fyij = fs * d.y;
                        fzij = fs * d.z;

                        fij_v += qB[j] * vij;
                        fij_x += qB[j] * fxij;
                        fij_y += qB[j] * fyij;
                        fij_z += qB[j] * fzij;
                    } // for j

                    // write back accumulated forces
                    fA[i].v = fij_v;
                    fA[i].x = fij_x;
                    fA[i].y = fij_y;
                    fA[i].z = fij_z;

                } // for i

            } // for k

        } // for l

#ifdef _OPENMP
    } // end parallel
#endif

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
