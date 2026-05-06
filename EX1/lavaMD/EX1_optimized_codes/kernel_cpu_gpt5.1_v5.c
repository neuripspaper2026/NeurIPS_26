#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h> // needed by malloc
#include <stdio.h>  // needed by printf
#include <math.h>   // needed by expf

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

    // counters
    long l;
    int k;

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
    THREE_VECTOR d;

    time1 = get_time();

    time2 = get_time();

    time3 = get_time();

    for (l = 0; l < dim.number_boxes; l++) {

        const box_str *const boxL = &box[l];

        first_i = boxL->offset; // offset to common arrays

        rA = &rv[first_i];
        fA = &fv[first_i];

        const int nn = boxL->nn;

        for (k = 0; k < (1 + nn); k++) {

            if (k == 0) {
                pointer = l; // set first box to be processed to home box
            } else {
                pointer = boxL->nei[k - 1].number; // remaining boxes are neighbor boxes
            }

            const box_str *const boxP = &box[pointer];

            first_j = boxP->offset;

            rB = &rv[first_j];
            qB = &qv[first_j];

            // perform manual unrolling on i and j loops for better ILP and reduced loop overhead

            int i = 0;
            for (; i <= NUMBER_PAR_PER_BOX - 2; i += 2) {

                FOUR_VECTOR *const rAi0 = &rA[i];
                FOUR_VECTOR *const rAi1 = &rA[i + 1];
                FOUR_VECTOR *const fAi0 = &fA[i];
                FOUR_VECTOR *const fAi1 = &fA[i + 1];

                fp fAi0_v = fAi0->v;
                fp fAi0_x = fAi0->x;
                fp fAi0_y = fAi0->y;
                fp fAi0_z = fAi0->z;

                fp fAi1_v = fAi1->v;
                fp fAi1_x = fAi1->x;
                fp fAi1_y = fAi1->y;
                fp fAi1_z = fAi1->z;

                int j = 0;
                for (; j <= NUMBER_PAR_PER_BOX - 4; j += 4) {

                    const FOUR_VECTOR *const rBj0 = &rB[j];
                    const FOUR_VECTOR *const rBj1 = &rB[j + 1];
                    const FOUR_VECTOR *const rBj2 = &rB[j + 2];
                    const FOUR_VECTOR *const rBj3 = &rB[j + 3];

                    const fp qBj0 = qB[j];
                    const fp qBj1 = qB[j + 1];
                    const fp qBj2 = qB[j + 2];
                    const fp qBj3 = qB[j + 3];

                    // i (first particle) interactions

                    // j
                    r2 = rAi0->v + rBj0->v - (rAi0->x * rBj0->x + rAi0->y * rBj0->y + rAi0->z * rBj0->z);
                    u2 = a2 * r2;
                    vij = expf(-u2);
                    fs = (fp)2.0 * vij;
                    d.x = rAi0->x - rBj0->x;
                    d.y = rAi0->y - rBj0->y;
                    d.z = rAi0->z - rBj0->z;
                    fAi0_v += qBj0 * vij;
                    fAi0_x += qBj0 * fs * d.x;
                    fAi0_y += qBj0 * fs * d.y;
                    fAi0_z += qBj0 * fs * d.z;

                    // j+1
                    r2 = rAi0->v + rBj1->v - (rAi0->x * rBj1->x + rAi0->y * rBj1->y + rAi0->z * rBj1->z);
                    u2 = a2 * r2;
                    vij = expf(-u2);
                    fs = (fp)2.0 * vij;
                    d.x = rAi0->x - rBj1->x;
                    d.y = rAi0->y - rBj1->y;
                    d.z = rAi0->z - rBj1->z;
                    fAi0_v += qBj1 * vij;
                    fAi0_x += qBj1 * fs * d.x;
                    fAi0_y += qBj1 * fs * d.y;
                    fAi0_z += qBj1 * fs * d.z;

                    // j+2
                    r2 = rAi0->v + rBj2->v - (rAi0->x * rBj2->x + rAi0->y * rBj2->y + rAi0->z * rBj2->z);
                    u2 = a2 * r2;
                    vij = expf(-u2);
                    fs = (fp)2.0 * vij;
                    d.x = rAi0->x - rBj2->x;
                    d.y = rAi0->y - rBj2->y;
                    d.z = rAi0->z - rBj2->z;
                    fAi0_v += qBj2 * vij;
                    fAi0_x += qBj2 * fs * d.x;
                    fAi0_y += qBj2 * fs * d.y;
                    fAi0_z += qBj2 * fs * d.z;

                    // j+3
                    r2 = rAi0->v + rBj3->v - (rAi0->x * rBj3->x + rAi0->y * rBj3->y + rAi0->z * rBj3->z);
                    u2 = a2 * r2;
                    vij = expf(-u2);
                    fs = (fp)2.0 * vij;
                    d.x = rAi0->x - rBj3->x;
                    d.y = rAi0->y - rBj3->y;
                    d.z = rAi0->z - rBj3->z;
                    fAi0_v += qBj3 * vij;
                    fAi0_x += qBj3 * fs * d.x;
                    fAi0_y += qBj3 * fs * d.y;
                    fAi0_z += qBj3 * fs * d.z;

                    // i+1 (second particle) interactions

                    // j
                    r2 = rAi1->v + rBj0->v - (rAi1->x * rBj0->x + rAi1->y * rBj0->y + rAi1->z * rBj0->z);
                    u2 = a2 * r2;
                    vij = expf(-u2);
                    fs = (fp)2.0 * vij;
                    d.x = rAi1->x - rBj0->x;
                    d.y = rAi1->y - rBj0->y;
                    d.z = rAi1->z - rBj0->z;
                    fAi1_v += qBj0 * vij;
                    fAi1_x += qBj0 * fs * d.x;
                    fAi1_y += qBj0 * fs * d.y;
                    fAi1_z += qBj0 * fs * d.z;

                    // j+1
                    r2 = rAi1->v + rBj1->v - (rAi1->x * rBj1->x + rAi1->y * rBj1->y + rAi1->z * rBj1->z);
                    u2 = a2 * r2;
                    vij = expf(-u2);
                    fs = (fp)2.0 * vij;
                    d.x = rAi1->x - rBj1->x;
                    d.y = rAi1->y - rBj1->y;
                    d.z = rAi1->z - rBj1->z;
                    fAi1_v += qBj1 * vij;
                    fAi1_x += qBj1 * fs * d.x;
                    fAi1_y += qBj1 * fs * d.y;
                    fAi1_z += qBj1 * fs * d.z;

                    // j+2
                    r2 = rAi1->v + rBj2->v - (rAi1->x * rBj2->x + rAi1->y * rBj2->y + rAi1->z * rBj2->z);
                    u2 = a2 * r2;
                    vij = expf(-u2);
                    fs = (fp)2.0 * vij;
                    d.x = rAi1->x - rBj2->x;
                    d.y = rAi1->y - rBj2->y;
                    d.z = rAi1->z - rBj2->z;
                    fAi1_v += qBj2 * vij;
                    fAi1_x += qBj2 * fs * d.x;
                    fAi1_y += qBj2 * fs * d.y;
                    fAi1_z += qBj2 * fs * d.z;

                    // j+3
                    r2 = rAi1->v + rBj3->v - (rAi1->x * rBj3->x + rAi1->y * rBj3->y + rAi1->z * rBj3->z);
                    u2 = a2 * r2;
                    vij = expf(-u2);
                    fs = (fp)2.0 * vij;
                    d.x = rAi1->x - rBj3->x;
                    d.y = rAi1->y - rBj3->y;
                    d.z = rAi1->z - rBj3->z;
                    fAi1_v += qBj3 * vij;
                    fAi1_x += qBj3 * fs * d.x;
                    fAi1_y += qBj3 * fs * d.y;
                    fAi1_z += qBj3 * fs * d.z;
                }

                // remaining j's for this pair of i,i+1
                for (; j < NUMBER_PAR_PER_BOX; j++) {

                    const FOUR_VECTOR *const rBj = &rB[j];
                    const fp qBj = qB[j];

                    // i
                    r2 = rAi0->v + rBj->v - (rAi0->x * rBj->x + rAi0->y * rBj->y + rAi0->z * rBj->z);
                    u2 = a2 * r2;
                    vij = expf(-u2);
                    fs = (fp)2.0 * vij;
                    d.x = rAi0->x - rBj->x;
                    d.y = rAi0->y - rBj->y;
                    d.z = rAi0->z - rBj->z;
                    fAi0_v += qBj * vij;
                    fAi0_x += qBj * fs * d.x;
                    fAi0_y += qBj * fs * d.y;
                    fAi0_z += qBj * fs * d.z;

                    // i+1
                    r2 = rAi1->v + rBj->v - (rAi1->x * rBj->x + rAi1->y * rBj->y + rAi1->z * rBj->z);
                    u2 = a2 * r2;
                    vij = expf(-u2);
                    fs = (fp)2.0 * vij;
                    d.x = rAi1->x - rBj->x;
                    d.y = rAi1->y - rBj->y;
                    d.z = rAi1->z - rBj->z;
                    fAi1_v += qBj * vij;
                    fAi1_x += qBj * fs * d.x;
                    fAi1_y += qBj * fs * d.y;
                    fAi1_z += qBj * fs * d.z;
                }

                fAi0->v = fAi0_v;
                fAi0->x = fAi0_x;
                fAi0->y = fAi0_y;
                fAi0->z = fAi0_z;

                fAi1->v = fAi1_v;
                fAi1->x = fAi1_x;
                fAi1->y = fAi1_y;
                fAi1->z = fAi1_z;
            }

            // handle remaining last i (if NUMBER_PAR_PER_BOX is odd)
            for (; i < NUMBER_PAR_PER_BOX; i++) {

                FOUR_VECTOR *const rAi = &rA[i];
                FOUR_VECTOR *const fAi = &fA[i];

                fp fAi_v = fAi->v;
                fp fAi_x = fAi->x;
                fp fAi_y = fAi->y;
                fp fAi_z = fAi->z;

                int j = 0;
                for (; j <= NUMBER_PAR_PER_BOX - 4; j += 4) {

                    const FOUR_VECTOR *const rBj0 = &rB[j];
                    const FOUR_VECTOR *const rBj1 = &rB[j + 1];
                    const FOUR_VECTOR *const rBj2 = &rB[j + 2];
                    const FOUR_VECTOR *const rBj3 = &rB[j + 3];

                    const fp qBj0 = qB[j];
                    const fp qBj1 = qB[j + 1];
                    const fp qBj2 = qB[j + 2];
                    const fp qBj3 = qB[j + 3];

                    // j
                    r2 = rAi->v + rBj0->v - (rAi->x * rBj0->x + rAi->y * rBj0->y + rAi->z * rBj0->z);
                    u2 = a2 * r2;
                    vij = expf(-u2);
                    fs = (fp)2.0 * vij;
                    d.x = rAi->x - rBj0->x;
                    d.y = rAi->y - rBj0->y;
                    d.z = rAi->z - rBj0->z;
                    fAi_v += qBj0 * vij;
                    fAi_x += qBj0 * fs * d.x;
                    fAi_y += qBj0 * fs * d.y;
                    fAi_z += qBj0 * fs * d.z;

                    // j+1
                    r2 = rAi->v + rBj1->v - (rAi->x * rBj1->x + rAi->y * rBj1->y + rAi->z * rBj1->z);
                    u2 = a2 * r2;
                    vij = expf(-u2);
                    fs = (fp)2.0 * vij;
                    d.x = rAi->x - rBj1->x;
                    d.y = rAi->y - rBj1->y;
                    d.z = rAi->z - rBj1->z;
                    fAi_v += qBj1 * vij;
                    fAi_x += qBj1 * fs * d.x;
                    fAi_y += qBj1 * fs * d.y;
                    fAi_z += qBj1 * fs * d.z;

                    // j+2
                    r2 = rAi->v + rBj2->v - (rAi->x * rBj2->x + rAi->y * rBj2->y + rAi->z * rBj2->z);
                    u2 = a2 * r2;
                    vij = expf(-u2);
                    fs = (fp)2.0 * vij;
                    d.x = rAi->x - rBj2->x;
                    d.y = rAi->y - rBj2->y;
                    d.z = rAi->z - rBj2->z;
                    fAi_v += qBj2 * vij;
                    fAi_x += qBj2 * fs * d.x;
                    fAi_y += qBj2 * fs * d.y;
                    fAi_z += qBj2 * fs * d.z;

                    // j+3
                    r2 = rAi->v + rBj3->v - (rAi->x * rBj3->x + rAi->y * rBj3->y + rAi->z * rBj3->z);
                    u2 = a2 * r2;
                    vij = expf(-u2);
                    fs = (fp)2.0 * vij;
                    d.x = rAi->x - rBj3->x;
                    d.y = rAi->y - rBj3->y;
                    d.z = rAi->z - rBj3->z;
                    fAi_v += qBj3 * vij;
                    fAi_x += qBj3 * fs * d.x;
                    fAi_y += qBj3 * fs * d.y;
                    fAi_z += qBj3 * fs * d.z;
                }

                // remaining j's
                for (; j < NUMBER_PAR_PER_BOX; j++) {

                    const FOUR_VECTOR *const rBj = &rB[j];
                    const fp qBj = qB[j];

                    r2 = rAi->v + rBj->v - (rAi->x * rBj->x + rAi->y * rBj->y + rAi->z * rBj->z);
                    u2 = a2 * r2;
                    vij = expf(-u2);
                    fs = (fp)2.0 * vij;
                    d.x = rAi->x - rBj->x;
                    d.y = rAi->y - rBj->y;
                    d.z = rAi->z - rBj->z;
                    fAi_v += qBj * vij;
                    fAi_x += qBj * fs * d.x;
                    fAi_y += qBj * fs * d.y;
                    fAi_z += qBj * fs * d.z;
                }

                fAi->v = fAi_v;
                fAi->x = fAi_x;
                fAi->y = fAi_y;
                fAi->z = fAi_z;
            }

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
