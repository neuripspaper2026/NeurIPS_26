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

    for (l = 0; l < dim.number_boxes; l++) {

        first_i = box[l].offset; // offset to common arrays

        rA = &rv[first_i];
        fA = &fv[first_i];

        const int nn = box[l].nn;

        for (k = 0; k <= nn; k++) {

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

                // cache home particle data to registers
                const fp rAix = rA[i].x;
                const fp rAiy = rA[i].y;
                const fp rAiz = rA[i].z;
                const fp rAiv = rA[i].v;

                fp fAv = fA[i].v;
                fp fAx = fA[i].x;
                fp fAy = fA[i].y;
                fp fAz = fA[i].z;

                // do for the # of particles in current (home or neighbor) box
                for (j = 0; j < NUMBER_PAR_PER_BOX; j++) {

                    const fp rBjx = rB[j].x;
                    const fp rBjy = rB[j].y;
                    const fp rBjz = rB[j].z;
                    const fp rBjv = rB[j].v;
                    const fp qBj  = qB[j];

                    // // coefficients
                    r2 = rAiv + rBjv - (rAix * rBjx + rAiy * rBjy + rAiz * rBjz);
                    u2 = a2 * r2;
                    vij = expf(-u2);
                    fs = (fp)2.0 * vij;
                    d.x = rAix - rBjx;
                    d.y = rAiy - rBjy;
                    d.z = rAiz - rBjz;
                    fxij = fs * d.x;
                    fyij = fs * d.y;
                    fzij = fs * d.z;

                    // forces
                    fAv += qBj * vij;
                    fAx += qBj * fxij;
                    fAy += qBj * fyij;
                    fAz += qBj * fzij;

                } // for j

                // write back accumulated forces once per i
                fA[i].v = fAv;
                fA[i].x = fAx;
                fA[i].y = fAy;
                fA[i].z = fAz;

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
