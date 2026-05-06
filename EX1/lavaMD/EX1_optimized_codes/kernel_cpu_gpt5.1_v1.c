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

    long long time0, time1, time2, time3, time4;

    fp alpha = par.alpha;
    fp a2 = (fp)2.0 * alpha * alpha;

    int l, k;

    time0 = get_time();
    time1 = get_time();
    time2 = get_time();
    time3 = get_time();

    for (l = 0; l < dim.number_boxes; ++l) {

        long first_i = box[l].offset;

        FOUR_VECTOR *const rA = &rv[first_i];
        FOUR_VECTOR *const fA = &fv[first_i];

        for (k = 0; k <= box[l].nn; ++k) {

            const int pointer = (k == 0) ? l : box[l].nei[k - 1].number;

            long first_j = box[pointer].offset;

            FOUR_VECTOR *const rB = &rv[first_j];
            fp *const qB = &qv[first_j];

            for (int i = 0; i < NUMBER_PAR_PER_BOX; ++i) {

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

                for (int j = 0; j < NUMBER_PAR_PER_BOX; ++j) {

                    const FOUR_VECTOR rBj = rB[j];
                    const fp qBj = qB[j];

                    const fp dx = rAi_x - rBj.x;
                    const fp dy = rAi_y - rBj.y;
                    const fp dz = rAi_z - rBj.z;

                    const fp r2 = rAi_v + rBj.v - (rAi_x * rBj.x + rAi_y * rBj.y + rAi_z * rBj.z);
                    const fp u2 = a2 * r2;
                    const fp vij = expf(-u2);
                    const fp fs = (fp)2.0 * vij;

                    const fp fxij = fs * dx;
                    const fp fyij = fs * dy;
                    const fp fzij = fs * dz;

                    const fp qvij = qBj * vij;

                    fAi_v += qvij;
                    fAi_x += qBj * fxij;
                    fAi_y += qBj * fyij;
                    fAi_z += qBj * fzij;
                }

                fA[i].v = fAi_v;
                fA[i].x = fAi_x;
                fA[i].y = fAi_y;
                fA[i].z = fAi_z;
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
