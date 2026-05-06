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
    fp alpha;
    fp a2;

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

    alpha = par.alpha;
    a2 = (fp)(2.0) * alpha * alpha;

    time3 = get_time();

    // Parallelize over boxes; each thread works on disjoint box ranges.
    // Inner loops are unchanged to preserve numerical behavior.
    #pragma omp parallel for default(none) private(l, first_i, rA, fA, k, pointer, first_j, rB, qB, i, j, r2, u2, fs, vij, d, fxij, fyij, fzij) shared(dim, box, rv, fv, qv, a2)
    for (l = 0; l < dim.number_boxes; l++) {

        first_i = box[l].offset; // offset to common arrays

        rA = &rv[first_i];
        fA = &fv[first_i];

        // process home box and its neighbor boxes
        for (k = 0; k < (1 + box[l].nn); k++) {

            if (k == 0) {
                pointer = l; // set first box to be processed to home box
            } else {
                pointer = box[l].nei[k - 1].number; // remaining boxes are neighbor boxes
            }

            first_j = box[pointer].offset;

            rB = &rv[first_j];
            qB = &qv[first_j];

            // manually unrolled inner loops for better ILP and vectorization
            for (i = 0; i < NUMBER_PAR_PER_BOX; i++) {

                FOUR_VECTOR *const rAi = &rA[i];
                FOUR_VECTOR *const fAi = &fA[i];

                int j_lim = NUMBER_PAR_PER_BOX & ~3; // multiple of 4
                int j_idx = 0;

                // process 4 interactions per iteration
                for (j = 0; j < j_lim; j += 4) {

                    // interaction j
                    {
                        const FOUR_VECTOR *const rBj = &rB[j];
                        const fp qBj = qB[j];

                        r2 = rAi->v + rBj->v - DOT((*rAi), (*rBj));
                        u2 = a2 * r2;
                        vij = expf(-u2);
                        fs = (fp)2.0 * vij;
                        d.x = rAi->x - rBj->x;
                        d.y = rAi->y - rBj->y;
                        d.z = rAi->z - rBj->z;
                        fxij = fs * d.x;
                        fyij = fs * d.y;
                        fzij = fs * d.z;

                        fAi->v += qBj * vij;
                        fAi->x += qBj * fxij;
                        fAi->y += qBj * fyij;
                        fAi->z += qBj * fzij;
                    }

                    // interaction j+1
                    {
                        const FOUR_VECTOR *const rBj = &rB[j + 1];
                        const fp qBj = qB[j + 1];

                        r2 = rAi->v + rBj->v - DOT((*rAi), (*rBj));
                        u2 = a2 * r2;
                        vij = expf(-u2);
                        fs = (fp)2.0 * vij;
                        d.x = rAi->x - rBj->x;
                        d.y = rAi->y - rBj->y;
                        d.z = rAi->z - rBj->z;
                        fxij = fs * d.x;
                        fyij = fs * d.y;
                        fzij = fs * d.z;

                        fAi->v += qBj * vij;
                        fAi->x += qBj * fxij;
                        fAi->y += qBj * fyij;
                        fAi->z += qBj * fzij;
                    }

                    // interaction j+2
                    {
                        const FOUR_VECTOR *const rBj = &rB[j + 2];
                        const fp qBj = qB[j + 2];

                        r2 = rAi->v + rBj->v - DOT((*rAi), (*rBj));
                        u2 = a2 * r2;
                        vij = expf(-u2);
                        fs = (fp)2.0 * vij;
                        d.x = rAi->x - rBj->x;
                        d.y = rAi->y - rBj->y;
                        d.z = rAi->z - rBj->z;
                        fxij = fs * d.x;
                        fyij = fs * d.y;
                        fzij = fs * d.z;

                        fAi->v += qBj * vij;
                        fAi->x += qBj * fxij;
                        fAi->y += qBj * fyij;
                        fAi->z += qBj * fzij;
                    }

                    // interaction j+3
                    {
                        const FOUR_VECTOR *const rBj = &rB[j + 3];
                        const fp qBj = qB[j + 3];

                        r2 = rAi->v + rBj->v - DOT((*rAi), (*rBj));
                        u2 = a2 * r2;
                        vij = expf(-u2);
                        fs = (fp)2.0 * vij;
                        d.x = rAi->x - rBj->x;
                        d.y = rAi->y - rBj->y;
                        d.z = rAi->z - rBj->z;
                        fxij = fs * d.x;
                        fyij = fs * d.y;
                        fzij = fs * d.z;

                        fAi->v += qBj * vij;
                        fAi->x += qBj * fxij;
                        fAi->y += qBj * fyij;
                        fAi->z += qBj * fzij;
                    }

                    j_idx = j + 4;
                }

                // remainder loop (if NUMBER_PAR_PER_BOX not divisible by 4)
                for (j = j_idx; j < NUMBER_PAR_PER_BOX; j++) {

                    const FOUR_VECTOR *const rBj = &rB[j];
                    const fp qBj = qB[j];

                    r2 = rAi->v + rBj->v - DOT((*rAi), (*rBj));
                    u2 = a2 * r2;
                    vij = expf(-u2);
                    fs = (fp)2.0 * vij;
                    d.x = rAi->x - rBj->x;
                    d.y = rAi->y - rBj->y;
                    d.z = rAi->z - rBj->z;
                    fxij = fs * d.x;
                    fyij = fs * d.y;
                    fzij = fs * d.z;

                    fAi->v += qBj * vij;
                    fAi->x += qBj * fxij;
                    fAi->y += qBj * fyij;
                    fAi->z += qBj * fzij;
                }

            } // for i

        } // for k

    } // for l

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
