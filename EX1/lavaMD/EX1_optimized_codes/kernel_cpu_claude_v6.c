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
    time0 = get_time();

    long long time1;
    long long time2;
    long long time3;
    long long time4;

    fp alpha;
    fp a2;

    int i, j, k, l;

    long first_i;
    FOUR_VECTOR *rA;
    FOUR_VECTOR *fA;

    int pointer;
    long first_j;
    FOUR_VECTOR *rB;
    fp *qB;

    fp r2;
    fp u2;
    fp fs;
    fp vij;
    fp fxij, fyij, fzij;
    THREE_VECTOR d;

    time1 = get_time();
    time2 = get_time();

    alpha = par.alpha;
    a2 = 2.0 * alpha * alpha;

    time3 = get_time();

    for (l = 0; l < dim.number_boxes; l++) {

        first_i = box[l].offset;

        rA = &rv[first_i];
        fA = &fv[first_i];

        int nn = box[l].nn;

        for (k = 0; k < (1 + nn); k++) {

            if (k == 0) {
                pointer = l;
            } else {
                pointer = box[l].nei[k - 1].number;
            }

            first_j = box[pointer].offset;

            rB = &rv[first_j];
            qB = &qv[first_j];

            for (i = 0; i < NUMBER_PAR_PER_BOX; i++) {

                fp fA_i_v = fA[i].v;
                fp fA_i_x = fA[i].x;
                fp fA_i_y = fA[i].y;
                fp fA_i_z = fA[i].z;

                fp rA_i_v = rA[i].v;
                fp rA_i_x = rA[i].x;
                fp rA_i_y = rA[i].y;
                fp rA_i_z = rA[i].z;

                for (j = 0; j < NUMBER_PAR_PER_BOX; j++) {

                    fp rB_j_v = rB[j].v;
                    fp rB_j_x = rB[j].x;
                    fp rB_j_y = rB[j].y;
                    fp rB_j_z = rB[j].z;
                    fp qB_j = qB[j];

                    r2 = rA_i_v + rB_j_v - (rA_i_x * rB_j_x + rA_i_y * rB_j_y + rA_i_z * rB_j_z);
                    u2 = a2 * r2;
                    vij = exp(-u2);
                    fs = 2.0 * vij;
                    d.x = rA_i_x - rB_j_x;
                    d.y = rA_i_y - rB_j_y;
                    d.z = rA_i_z - rB_j_z;
                    fxij = fs * d.x;
                    fyij = fs * d.y;
                    fzij = fs * d.z;

                    fA_i_v += qB_j * vij;
                    fA_i_x += qB_j * fxij;
                    fA_i_y += qB_j * fyij;
                    fA_i_z += qB_j * fzij;
                }

                fA[i].v = fA_i_v;
                fA[i].x = fA_i_x;
                fA[i].y = fA_i_y;
                fA[i].z = fA_i_z;
            }
        }
    }

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
}

#ifdef __cplusplus
}
#endif
