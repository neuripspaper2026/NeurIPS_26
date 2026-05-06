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
    int k, i, j;

    // home box
    long first_i;
    FOUR_VECTOR *rA;
    FOUR_VECTOR *fA;

    // neighbor box
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

    for (l = 0; l < dim.number_boxes; ++l) {

        const box_str *const box_l = &box[l];
        first_i = box_l->offset; // offset to common arrays

        rA = &rv[first_i];
        fA = &fv[first_i];

        const int nn = box_l->nn;

        for (k = 0; k <= nn; ++k) {

            const int pointer = (k == 0) ? l : box_l->nei[k - 1].number;

            const box_str *const box_p = &box[pointer];
            first_j = box_p->offset;

            rB = &rv[first_j];
            qB = &qv[first_j];

            for (i = 0; i < NUMBER_PAR_PER_BOX; ++i) {

                FOUR_VECTOR *const fAi = &fA[i];
                const FOUR_VECTOR *const rAi = &rA[i];
                const fp rAi_v = rAi->v;
                const fp rAi_x = rAi->x;
                const fp rAi_y = rAi->y;
                const fp rAi_z = rAi->z;

                fp acc_v = fAi->v;
                fp acc_x = fAi->x;
                fp acc_y = fAi->y;
                fp acc_z = fAi->z;

                for (j = 0; j < NUMBER_PAR_PER_BOX; ++j) {

                    const FOUR_VECTOR *const rBj = &rB[j];
                    const fp qBj = qB[j];

                    // coefficients
                    r2 = rAi_v + rBj->v -
                         (rAi_x * rBj->x + rAi_y * rBj->y + rAi_z * rBj->z);
                    u2 = a2 * r2;
                    vij = expf(-u2);
                    fs = (fp)2.0 * vij;

                    d.x = rAi_x - rBj->x;
                    d.y = rAi_y - rBj->y;
                    d.z = rAi_z - rBj->z;

                    fxij = fs * d.x;
                    fyij = fs * d.y;
                    fzij = fs * d.z;

                    // accumulate forces
                    const fp qv_vij = qBj * vij;
                    acc_v += qv_vij;
                    acc_x += qBj * fxij;
                    acc_y += qBj * fyij;
                    acc_z += qBj * fzij;

                } // for j

                fAi->v = acc_v;
                fAi->x = acc_x;
                fAi->y = acc_y;
                fAi->z = acc_z;

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
