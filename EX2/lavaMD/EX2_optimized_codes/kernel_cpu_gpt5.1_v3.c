#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h> // (in path known to compiler)			needed by malloc
#include <stdio.h>  // (in path known to compiler)			needed by printf
#include <math.h>   // (in path known to compiler)			needed by exp
#ifdef _OPENMP
#include <omp.h>
#endif

#include "../lavaMD.h" // (in the main program folder)	needed to recognized input variables

#include "../util/timer/timer.h" // (in library path specified to compiler)	needed by timer

#include "kernel_cpu.h" // (in the current directory)

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

    time1 = get_time();

    time2 = get_time();

    time3 = get_time();

    // main computation
#ifdef _OPENMP
#pragma omp parallel
    {
        int l, k, i, j;

        long first_i;
        FOUR_VECTOR *rA;
        FOUR_VECTOR *fA;

        int pointer;
        long first_j;
        FOUR_VECTOR *rB;
        fp *qB;

        fp r2, u2;
        fp fs;
        fp vij;
        fp fxij, fyij, fzij;

        // private temporaries to help compiler vectorize
        fp rAx, rAy, rAz, rAv;
        fp rBx, rBy, rBz, rBv;
        THREE_VECTOR d;

#pragma omp for schedule(static) nowait
        for (l = 0; l < dim.number_boxes; l++) {

            first_i = box[l].offset; // offset to common arrays

            rA = &rv[first_i];
            fA = &fv[first_i];

            for (k = 0; k < (1 + box[l].nn); k++) {

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

                    rAx = rA[i].x;
                    rAy = rA[i].y;
                    rAz = rA[i].z;
                    rAv = rA[i].v;

                    // accumulate into locals to reduce memory traffic
                    fp fA_v = fA[i].v;
                    fp fA_x = fA[i].x;
                    fp fA_y = fA[i].y;
                    fp fA_z = fA[i].z;

#pragma omp simd private(j, rBx, rBy, rBz, rBv, d, r2, u2, vij, fs, fxij, fyij, fzij) reduction(+:fA_v,fA_x,fA_y,fA_z)
                    for (j = 0; j < NUMBER_PAR_PER_BOX; j++) {

                        rBx = rB[j].x;
                        rBy = rB[j].y;
                        rBz = rB[j].z;
                        rBv = rB[j].v;

                        r2 = rAv + rBv - (rAx * rBx + rAy * rBy + rAz * rBz);
                        u2 = a2 * r2;
                        vij = expf(-u2);
                        fs = (fp)2.0 * vij;

                        d.x = rAx - rBx;
                        d.y = rAy - rBy;
                        d.z = rAz - rBz;

                        fxij = fs * d.x;
                        fyij = fs * d.y;
                        fzij = fs * d.z;

                        fA_v += qB[j] * vij;
                        fA_x += qB[j] * fxij;
                        fA_y += qB[j] * fyij;
                        fA_z += qB[j] * fzij;

                    } // for j

                    fA[i].v = fA_v;
                    fA[i].x = fA_x;
                    fA[i].y = fA_y;
                    fA[i].z = fA_z;

                } // for i

            } // for k

        } // for l
    }     // omp parallel
#else
    {
        int l, k, i, j;

        long first_i;
        FOUR_VECTOR *rA;
        FOUR_VECTOR *fA;

        int pointer;
        long first_j;
        FOUR_VECTOR *rB;
        fp *qB;

        fp r2, u2;
        fp fs;
        fp vij;
        fp fxij, fyij, fzij;

        fp rAx, rAy, rAz, rAv;
        fp rBx, rBy, rBz, rBv;
        THREE_VECTOR d;

        for (l = 0; l < dim.number_boxes; l++) {

            first_i = box[l].offset; // offset to common arrays

            rA = &rv[first_i];
            fA = &fv[first_i];

            for (k = 0; k < (1 + box[l].nn); k++) {

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

                    rAx = rA[i].x;
                    rAy = rA[i].y;
                    rAz = rA[i].z;
                    rAv = rA[i].v;

                    fp fA_v = fA[i].v;
                    fp fA_x = fA[i].x;
                    fp fA_y = fA[i].y;
                    fp fA_z = fA[i].z;

#pragma omp simd private(j, rBx, rBy, rBz, rBv, d, r2, u2, vij, fs, fxij, fyij, fzij) reduction(+:fA_v,fA_x,fA_y,fA_z)
                    for (j = 0; j < NUMBER_PAR_PER_BOX; j++) {

                        rBx = rB[j].x;
                        rBy = rB[j].y;
                        rBz = rB[j].z;
                        rBv = rB[j].v;

                        r2 = rAv + rBv - (rAx * rBx + rAy * rBy + rAz * rBz);
                        u2 = a2 * r2;
                        vij = expf(-u2);
                        fs = (fp)2.0 * vij;

                        d.x = rAx - rBx;
                        d.y = rAy - rBy;
                        d.z = rAz - rBz;

                        fxij = fs * d.x;
                        fyij = fs * d.y;
                        fzij = fs * d.z;

                        fA_v += qB[j] * vij;
                        fA_x += qB[j] * fxij;
                        fA_y += qB[j] * fyij;
                        fA_z += qB[j] * fzij;

                    } // for j

                    fA[i].v = fA_v;
                    fA[i].x = fA_x;
                    fA[i].y = fA_y;
                    fA[i].z = fA_z;

                } // for i

            } // for k

        } // for l
    }
#endif

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

} // main

#ifdef __cplusplus
}
#endif
