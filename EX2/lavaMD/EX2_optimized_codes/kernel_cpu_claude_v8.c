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

            fp local_fA_v[NUMBER_PAR_PER_BOX];
            fp local_fA_x[NUMBER_PAR_PER_BOX];
            fp local_fA_y[NUMBER_PAR_PER_BOX];
            fp local_fA_z[NUMBER_PAR_PER_BOX];

            for (int i = 0; i < NUMBER_PAR_PER_BOX; i++) {
                local_fA_v[i] = 0.0f;
                local_fA_x[i] = 0.0f;
                local_fA_y[i] = 0.0f;
                local_fA_z[i] = 0.0f;
            }

            for (int k = 0; k < (1 + box[l].nn); k++) {
                int pointer = (k == 0) ? l : box[l].nei[k - 1].number;
                long first_j = box[pointer].offset;
                FOUR_VECTOR *rB = &rv[first_j];
                fp *qB = &qv[first_j];

                for (int i = 0; i < NUMBER_PAR_PER_BOX; i++) {
                    fp rA_v = rA[i].v;
                    fp rA_x = rA[i].x;
                    fp rA_y = rA[i].y;
                    fp rA_z = rA[i].z;

                    fp acc_v = 0.0f;
                    fp acc_x = 0.0f;
                    fp acc_y = 0.0f;
                    fp acc_z = 0.0f;

                    #pragma omp simd reduction(+:acc_v,acc_x,acc_y,acc_z)
                    for (int j = 0; j < NUMBER_PAR_PER_BOX; j++) {
                        fp dx = rA_x - rB[j].x;
                        fp dy = rA_y - rB[j].y;
                        fp dz = rA_z - rB[j].z;

                        fp r2 = rA_v + rB[j].v - (rA_x * rB[j].x + rA_y * rB[j].y + rA_z * rB[j].z);
                        fp u2 = a2 * r2;
                        fp vij = expf(-u2);
                        fp fs = 2.0f * vij;

                        fp qB_j = qB[j];
                        acc_v += qB_j * vij;
                        acc_x += qB_j * fs * dx;
                        acc_y += qB_j * fs * dy;
                        acc_z += qB_j * fs * dz;
                    }

                    local_fA_v[i] += acc_v;
                    local_fA_x[i] += acc_x;
                    local_fA_y[i] += acc_y;
                    local_fA_z[i] += acc_z;
                }
            }

            for (int i = 0; i < NUMBER_PAR_PER_BOX; i++) {
                #pragma omp atomic
                fA[i].v += local_fA_v[i];
                #pragma omp atomic
                fA[i].x += local_fA_x[i];
                #pragma omp atomic
                fA[i].y += local_fA_y[i];
                #pragma omp atomic
                fA[i].z += local_fA_z[i];
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
