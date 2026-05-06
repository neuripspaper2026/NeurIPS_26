#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

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

    time1 = get_time();
    time2 = get_time();

    alpha = par.alpha;
    a2 = 2.0 * alpha * alpha;

    time3 = get_time();

    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, 1) nowait
        for (long l = 0; l < dim.number_boxes; l++) {

            long first_i = box[l].offset;

            FOUR_VECTOR *rA = &rv[first_i];
            FOUR_VECTOR *fA = &fv[first_i];

            FOUR_VECTOR fA_local[NUMBER_PAR_PER_BOX];
            for (int i = 0; i < NUMBER_PAR_PER_BOX; i++) {
                fA_local[i].v = 0.0f;
                fA_local[i].x = 0.0f;
                fA_local[i].y = 0.0f;
                fA_local[i].z = 0.0f;
            }

            for (int k = 0; k < (1 + box[l].nn); k++) {

                int pointer;
                if (k == 0) {
                    pointer = l;
                } else {
                    pointer = box[l].nei[k - 1].number;
                }

                long first_j = box[pointer].offset;

                FOUR_VECTOR *rB = &rv[first_j];
                fp *qB = &qv[first_j];

                for (int i = 0; i < NUMBER_PAR_PER_BOX; i++) {

                    fp rAi_v = rA[i].v;
                    fp rAi_x = rA[i].x;
                    fp rAi_y = rA[i].y;
                    fp rAi_z = rA[i].z;

                    fp fA_v_acc = 0.0f;
                    fp fA_x_acc = 0.0f;
                    fp fA_y_acc = 0.0f;
                    fp fA_z_acc = 0.0f;

                    #pragma omp simd reduction(+:fA_v_acc,fA_x_acc,fA_y_acc,fA_z_acc)
                    for (int j = 0; j < NUMBER_PAR_PER_BOX; j++) {

                        fp r2 = rAi_v + rB[j].v - (rAi_x * rB[j].x + rAi_y * rB[j].y + rAi_z * rB[j].z);
                        fp u2 = a2 * r2;
                        fp vij = expf(-u2);
                        fp fs = 2.0f * vij;
                        fp dx = rAi_x - rB[j].x;
                        fp dy = rAi_y - rB[j].y;
                        fp dz = rAi_z - rB[j].z;
                        fp fxij = fs * dx;
                        fp fyij = fs * dy;
                        fp fzij = fs * dz;

                        fp qBj = qB[j];
                        fA_v_acc += qBj * vij;
                        fA_x_acc += qBj * fxij;
                        fA_y_acc += qBj * fyij;
                        fA_z_acc += qBj * fzij;
                    }

                    fA_local[i].v += fA_v_acc;
                    fA_local[i].x += fA_x_acc;
                    fA_local[i].y += fA_y_acc;
                    fA_local[i].z += fA_z_acc;
                }
            }

            for (int i = 0; i < NUMBER_PAR_PER_BOX; i++) {
                #pragma omp atomic
                fA[i].v += fA_local[i].v;
                #pragma omp atomic
                fA[i].x += fA_local[i].x;
                #pragma omp atomic
                fA[i].y += fA_local[i].y;
                #pragma omp atomic
                fA[i].z += fA_local[i].z;
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
