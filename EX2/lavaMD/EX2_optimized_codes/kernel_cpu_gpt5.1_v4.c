#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h> // needed by malloc
#include <stdio.h>  // needed by printf
#include <math.h>   // needed by exp
#ifdef _OPENMP
#include <omp.h>
#endif

#include "../lavaMD.h" // needed to recognized input variables

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
    int i, j, k;

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

    // main computation
    // Parallelize outer loop over boxes; each box writes to a disjoint
    // segment of fv, so no cross-iteration races.
#ifdef _OPENMP
#pragma omp parallel for default(none) private(i, j, k, first_i, rA, fA, pointer, first_j, rB, qB, r2, u2, fs, vij, fxij, fyij, fzij, d) shared(dim, box, rv, fv, qv, a2)
#endif
    for (int l = 0; l < dim.number_boxes; l++) {

        first_i = box[l].offset; // offset to common arrays
        rA = &rv[first_i];
        fA = &fv[first_i];

        // Each (l,k) pair reads a neighbor box and updates only fA (home box),
        // so still no conflicts between different l iterations.
        for (k = 0; k < (1 + box[l].nn); k++) {

            pointer = (k == 0) ? l : box[l].nei[k - 1].number;

            first_j = box[pointer].offset;

            rB = &rv[first_j];
            qB = &qv[first_j];

            // Flatten i,j loops to help vectorization and improve locality.
            for (int ij = 0; ij < NUMBER_PAR_PER_BOX * NUMBER_PAR_PER_BOX; ++ij) {

                i = ij / NUMBER_PAR_PER_BOX;
                j = ij - i * NUMBER_PAR_PER_BOX;

                // coefficients
                r2 = rA[i].v + rB[j].v - DOT(rA[i], rB[j]);
                u2 = a2 * r2;
                vij = expf(-u2);
                fs = (fp)2.0 * vij;

                d.x = rA[i].x - rB[j].x;
                d.y = rA[i].y - rB[j].y;
                d.z = rA[i].z - rB[j].z;

                fxij = fs * d.x;
                fyij = fs * d.y;
                fzij = fs * d.z;

                // forces
                const fp qBj = qB[j];
                fA[i].v += qBj * vij;
                fA[i].x += qBj * fxij;
                fA[i].y += qBj * fyij;
                fA[i].z += qBj * fzij;

            } // for ij

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
