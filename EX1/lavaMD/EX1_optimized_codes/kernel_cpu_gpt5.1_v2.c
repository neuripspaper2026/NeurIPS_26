#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h> // (in path known to compiler)			needed by malloc
#include <stdio.h>  // (in path known to compiler)			needed by printf
#include <math.h>   // (in path known to compiler)			needed by exp

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

    // counters
    int i, j, k, l;

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
    fp fxij, fyij, fzij;
    THREE_VECTOR d;

    time1 = get_time();

    time2 = get_time();

    time3 = get_time();

    const int nboxes = (int)dim.number_boxes;

    for (l = 0; l < nboxes; ++l) {

        first_i = box[l].offset; // offset to common arrays

        rA = &rv[first_i];
        fA = &fv[first_i];

        const int n_par_i = NUMBER_PAR_PER_BOX;
        const int nnei = box[l].nn;

        for (k = 0; k < (1 + nnei); ++k) {

            if (k == 0) {
                pointer = l; // set first box to be processed to home box
            } else {
                pointer = box[l].nei[k - 1].number; // remaining boxes are neighbor boxes
            }

            first_j = box[pointer].offset;

            rB = &rv[first_j];
            qB = &qv[first_j];

            const int n_par_j = NUMBER_PAR_PER_BOX;

            for (i = 0; i < n_par_i; ++i) {

                FOUR_VECTOR rAi = rA[i];
                FOUR_VECTOR fAi = fA[i];
                const fp rAi_v = rAi.v;

                for (j = 0; j < n_par_j; ++j) {

                    const FOUR_VECTOR rBj = rB[j];

                    // coefficients
                    r2 = rAi_v + rBj.v - DOT(rAi, rBj);
                    u2 = a2 * r2;
                    vij = expf(-u2);
                    fs = (fp)2.0 * vij;
                    d.x = rAi.x - rBj.x;
                    d.y = rAi.y - rBj.y;
                    d.z = rAi.z - rBj.z;
                    fxij = fs * d.x;
                    fyij = fs * d.y;
                    fzij = fs * d.z;

                    // forces
                    const fp qBj = qB[j];
                    fAi.v += qBj * vij;
                    fAi.x += qBj * fxij;
                    fAi.y += qBj * fyij;
                    fAi.z += qBj * fzij;

                } // for j

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
