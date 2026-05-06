#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../lavaMD.h"
#include "../util/timer/timer.h"
#include "kernel_cpu.h"

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
    int k, i;

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
    fp vij;
    fp fs;

    time1 = get_time();

    time2 = get_time();

    time3 = get_time();

    for (l = 0; l < dim.number_boxes; l++) {

        first_i = box[l].offset;

        rA = &rv[first_i];
        fA = &fv[first_i];

        for (k = 0; k < box[l].nn + 1; k++) {

            if (k == 0) {
                pointer = l;
            } else {
                pointer = box[l].nei[k - 1].number;
            }

            first_j = box[pointer].offset;

            rB = &rv[first_j];
            qB = &qv[first_j];

            for (i = 0; i < NUMBER_PAR_PER_BOX; i++) {

                FOUR_VECTOR rAi = rA[i];
                FOUR_VECTOR fAi = fA[i];

                const fp rAi_v = rAi.v;
                const fp rAi_x = rAi.x;
                const fp rAi_y = rAi.y;
                const fp rAi_z = rAi.z;

                fp fAi_v = fAi.v;
                fp fAi_x = fAi.x;
                fp fAi_y = fAi.y;
                fp fAi_z = fAi.z;

                int j = 0;

#if NUMBER_PAR_PER_BOX % 4 != 0
#pragma GCC ivdep
#endif
                for (; j + 3 < NUMBER_PAR_PER_BOX; j += 4) {

                    // j
                    {
                        const FOUR_VECTOR rBj = rB[j];
                        const fp qBj = qB[j];

                        const fp dx = rAi_x - rBj.x;
                        const fp dy = rAi_y - rBj.y;
                        const fp dz = rAi_z - rBj.z;

                        r2 = rAi_v + rBj.v - (rAi_x * rBj.x + rAi_y * rBj.y + rAi_z * rBj.z);
                        u2 = a2 * r2;
                        vij = expf(-u2);
                        fs = (fp)2.0 * vij;

                        fAi_v += qBj * vij;
                        fAi_x += qBj * fs * dx;
                        fAi_y += qBj * fs * dy;
                        fAi_z += qBj * fs * dz;
                    }

                    // j + 1
                    {
                        const FOUR_VECTOR rBj = rB[j + 1];
                        const fp qBj = qB[j + 1];

                        const fp dx = rAi_x - rBj.x;
                        const fp dy = rAi_y - rBj.y;
                        const fp dz = rAi_z - rBj.z;

                        r2 = rAi_v + rBj.v - (rAi_x * rBj.x + rAi_y * rBj.y + rAi_z * rBj.z);
                        u2 = a2 * r2;
                        vij = expf(-u2);
                        fs = (fp)2.0 * vij;

                        fAi_v += qBj * vij;
                        fAi_x += qBj * fs * dx;
                        fAi_y += qBj * fs * dy;
                        fAi_z += qBj * fs * dz;
                    }

                    // j + 2
                    {
                        const FOUR_VECTOR rBj = rB[j + 2];
                        const fp qBj = qB[j + 2];

                        const fp dx = rAi_x - rBj.x;
                        const fp dy = rAi_y - rBj.y;
                        const fp dz = rAi_z - rBj.z;

                        r2 = rAi_v + rBj.v - (rAi_x * rBj.x + rAi_y * rBj.y + rAi_z * rBj.z);
                        u2 = a2 * r2;
                        vij = expf(-u2);
                        fs = (fp)2.0 * vij;

                        fAi_v += qBj * vij;
                        fAi_x += qBj * fs * dx;
                        fAi_y += qBj * fs * dy;
                        fAi_z += qBj * fs * dz;
                    }

                    // j + 3
                    {
                        const FOUR_VECTOR rBj = rB[j + 3];
                        const fp qBj = qB[j + 3];

                        const fp dx = rAi_x - rBj.x;
                        const fp dy = rAi_y - rBj.y;
                        const fp dz = rAi_z - rBj.z;

                        r2 = rAi_v + rBj.v - (rAi_x * rBj.x + rAi_y * rBj.y + rAi_z * rBj.z);
                        u2 = a2 * r2;
                        vij = expf(-u2);
                        fs = (fp)2.0 * vij;

                        fAi_v += qBj * vij;
                        fAi_x += qBj * fs * dx;
                        fAi_y += qBj * fs * dy;
                        fAi_z += qBj * fs * dz;
                    }
                }

                for (; j < NUMBER_PAR_PER_BOX; j++) {

                    const FOUR_VECTOR rBj = rB[j];
                    const fp qBj = qB[j];

                    const fp dx = rAi_x - rBj.x;
                    const fp dy = rAi_y - rBj.y;
                    const fp dz = rAi_z - rBj.z;

                    r2 = rAi_v + rBj.v - (rAi_x * rBj.x + rAi_y * rBj.y + rAi_z * rBj.z);
                    u2 = a2 * r2;
                    vij = expf(-u2);
                    fs = (fp)2.0 * vij;

                    fAi_v += qBj * vij;
                    fAi_x += qBj * fs * dx;
                    fAi_y += qBj * fs * dy;
                    fAi_z += qBj * fs * dz;
                }

                fAi.v = fAi_v;
                fAi.x = fAi_x;
                fAi.y = fAi_y;
                fAi.z = fAi_z;

                fA[i] = fAi;

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
