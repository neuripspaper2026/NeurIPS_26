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

    long long time0;
    long long time1;
    long long time2;
    long long time3;
    long long time4;

    const fp alpha = par.alpha;
    const fp a2 = (fp)2.0 * alpha * alpha;

    int l, k;

    time0 = get_time();
    time1 = get_time();
    time2 = get_time();
    time3 = get_time();

    for (l = 0; l < dim.number_boxes; ++l) {

        const long first_i = box[l].offset;
        FOUR_VECTOR *const rA = &rv[first_i];
        FOUR_VECTOR *const fA = &fv[first_i];

        for (k = 0; k < (1 + box[l].nn); ++k) {

            const int pointer = (k == 0) ? l : box[l].nei[k - 1].number;

            const long first_j = box[pointer].offset;
            FOUR_VECTOR *const rB = &rv[first_j];
            fp *const qB = &qv[first_j];

            for (int i = 0; i < NUMBER_PAR_PER_BOX; ++i) {

                const fp rAi_v = rA[i].v;
                const fp rAi_x = rA[i].x;
                const fp rAi_y = rA[i].y;
                const fp rAi_z = rA[i].z;

                fp fA_v = fA[i].v;
                fp fA_x = fA[i].x;
                fp fA_y = fA[i].y;
                fp fA_z = fA[i].z;

#pragma GCC ivdep
                for (int j = 0; j < NUMBER_PAR_PER_BOX; ++j) {

                    const fp rBj_v = rB[j].v;
                    const fp rBj_x = rB[j].x;
                    const fp rBj_y = rB[j].y;
                    const fp rBj_z = rB[j].z;
                    const fp qBj  = qB[j];

                    const fp dx = rAi_x - rBj_x;
                    const fp dy = rAi_y - rBj_y;
                    const fp dz = rAi_z - rBj_z;

                    const fp r2 = rAi_v + rBj_v - (rAi_x * rBj_x + rAi_y * rBj_y + rAi_z * rBj_z);
                    const fp u2 = a2 * r2;
                    const fp vij = expf(-u2);
                    const fp fs = (fp)2.0 * vij;

                    const fp fxij = fs * dx;
                    const fp fyij = fs * dy;
                    const fp fzij = fs * dz;

                    fA_v += qBj * vij;
                    fA_x += qBj * fxij;
                    fA_y += qBj * fyij;
                    fA_z += qBj * fzij;
                }

                fA[i].v = fA_v;
                fA[i].x = fA_x;
                fA[i].y = fA_y;
                fA[i].z = fA_z;

            }

        }

    }

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

}

#ifdef __cplusplus
}
#endif
