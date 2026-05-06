#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h> // needed by malloc
#include <stdio.h>  // needed by printf
#include <math.h>   // needed by expf

#include "../lavaMD.h" // needed to recognized input variables

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

    // counters
    int i, j, k, l;

    // home box
    long first_i;
    FOUR_VECTOR *rA;
    FOUR_VECTOR *fA;

    // neighbor box
    int pointer;
    long first_j;
    FOUR_VECTOR *rB;
    fp *qB;

    // common
    fp r2;
    fp u2;
    fp fs;
    fp vij;
    fp fxij, fyij, fzij;
    THREE_VECTOR d;

    time1 = get_time();

    time2 = get_time();

    time3 = get_time();

    for (l = 0; l < dim.number_boxes; l++) {

        first_i = box[l].offset; // offset to common arrays

        rA = &rv[first_i];
        fA = &fv[first_i];

        const int nn = box[l].nn;

        for (k = 0; k <= nn; k++) {

            if (k == 0) {
                pointer = l; // set first box to be processed to home box
            } else {
                pointer = box[l].nei[k - 1].number; // remaining boxes are
                                                    // neighbor boxes
            }

            first_j = box[pointer].offset;

            rB = &rv[first_j];
            qB = &qv[first_j];

            for (i = 0; i < NUMBER_PAR_PER_BOX; i++) {

                FOUR_VECTOR *const rAi = &rA[i];
                FOUR_VECTOR *const fAi = &fA[i];

                fp fAv = fAi->v;
                fp fAx = fAi->x;
                fp fAy = fAi->y;
                fp fAz = fAi->z;

                const fp rAiv = rAi->v;
                const fp rAix = rAi->x;
                const fp rAiy = rAi->y;
                const fp rAiz = rAi->z;

                for (j = 0; j < NUMBER_PAR_PER_BOX; j++) {

                    const FOUR_VECTOR *const rBj = &rB[j];
                    const fp qBj = qB[j];

                    // coefficients
                    r2 = rAiv + rBj->v -
                         (rAix * rBj->x + rAiy * rBj->y + rAiz * rBj->z);
                    u2 = a2 * r2;
                    vij = expf(-u2);
                    fs = (fp)2.0 * vij;
                    d.x = rAix - rBj->x;
                    d.y = rAiy - rBj->y;
                    d.z = rAiz - rBj->z;
                    fxij = fs * d.x;
                    fyij = fs * d.y;
                    fzij = fs * d.z;

                    // forces
                    fAv += qBj * vij;
                    fAx += qBj * fxij;
                    fAy += qBj * fyij;
                    fAz += qBj * fzij;

                } // for j

                fAi->v = fAv;
                fAi->x = fAx;
                fAi->y = fAy;
                fAi->z = fAz;

            } // for i

        } // for k

    } // for l

    time4 = get_time();

    printf("Time spent in different stages of CPU/MCPU KERNEL:\n");

    printf("%15.12f s, %15.12f % : CPU/MCPU: VARIABLES\n",
           (float)(time1 - time0) / 1000000.0f,
           (float)(time1 - time0) / (float)(time4 - time0) * 100.0f);
    printf("%15.12f s, %15.12f % : MCPU: SET DEVICE\n",
           (float)(time2 - time1) / 1000000.0f,
           (float)(time2 - time1) / (float)(time4 - time0) * 100.0f);
    printf("%15.12f s, %15.12f % : CPU/MCPU: INPUTS\n",
           (float)(time3 - time2) / 1000000.0f,
           (float)(time3 - time2) / (float)(time4 - time0) * 100.0f);
    printf("%15.12f s, %15.12f % : CPU/MCPU: KERNEL\n",
           (float)(time4 - time3) / 1000000.0f,
           (float)(time4 - time3) / (float)(time4 - time0) * 100.0f);

    printf("Total time:\n");
    printf("%.12f s\n", (float)(time4 - time0) / 1000000.0f);

} // main

#ifdef __cplusplus
}
#endif
