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

    long long time0 = get_time();
    long long time1, time2, time3, time4;

    fp alpha = par.alpha;
    fp a2 = 2.0f * alpha * alpha;

    time3 = get_time();

    for (int l = 0; l < dim.number_boxes; l++) {
        long first_i = box[l].offset;
        FOUR_VECTOR *rA = &rv[first_i];
        FOUR_VECTOR *fA = &fv[first_i];

        for (int k = 0; k < (1 + box[l].nn); k++) {
            int pointer = (k == 0) ? l : box[l].nei[k - 1].number;
            long first_j = box[pointer].offset;
            FOUR_VECTOR *rB = &rv[first_j];
            fp *qB = &qv[first_j];

            for (int i = 0; i < NUMBER_PAR_PER_BOX; i++) {
                FOUR_VECTOR ri = rA[i];
                FOUR_VECTOR *fi = &fA[i];

                for (int j = 0; j < NUMBER_PAR_PER_BOX; j++) {
                    fp rx = ri.x - rB[j].x;
                    fp ry = ri.y - rB[j].y;
                    fp rz = ri.z - rB[j].z;
                    
                    fp r2 = ri.v + rB[j].v - (rx * rB[j].x + ry * rB[j].y + rz * rB[j].z);
                    fp u2 = a2 * r2;
                    fp vij = expf(-u2);
                    fp fs = 2.0f * vij;
                    
                    fp fxij = fs * rx;
                    fp fyij = fs * ry;
                    fp fzij = fs * rz;

                    fi->v += qB[j] * vij;
                    fi->x += qB[j] * fxij;
                    fi->y += qB[j] * fyij;
                    fi->z += qB[j] * fzij;
                }
            }
        }
    }

    time4 = get_time();

    time1 = time0;
    time2 = time3;

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

}
#ifdef __cplusplus
}
#endif
