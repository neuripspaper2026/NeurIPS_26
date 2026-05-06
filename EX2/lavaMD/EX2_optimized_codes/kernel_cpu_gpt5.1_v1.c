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

    // parallelize over boxes; each box has a disjoint force array region
    // use static schedule to improve cache locality
#ifdef _OPENMP
#pragma omp parallel for default(none) private(l, k, pointer, first_i, first_j, rA, fA, rB, qB, i, j, \
                                               r2, u2, fs, vij, fxij, fyij, fzij, d)                    \
    shared(dim, box, rv, fv, qv, a2)
#endif
    for (l = 0; l < dim.number_boxes; l++) {

        first_i = box[l].offset; // offset to common arrays

        rA = &rv[first_i];
        fA = &fv[first_i];

        const int nn = box[l].nn;

        for (k = 0; k < (1 + nn); k++) {

            if (k == 0) {
                pointer = l; // set first box to be processed to home box
            } else {
                pointer = box[l].nei[k - 1].number; // remaining boxes are neighbor boxes
            }

            first_j = box[pointer].offset;

            rB = &rv[first_j];
            qB = &qv[first_j];

            // unroll inner j-loop for better ILP and vectorization
            for (i = 0; i < NUMBER_PAR_PER_BOX; i++) {

                const fp rA_v = rA[i].v;
                const fp rA_x = rA[i].x;
                const fp rA_y = rA[i].y;
                const fp rA_z = rA[i].z;

                fp fA_v = fA[i].v;
                fp fA_x = fA[i].x;
                fp fA_y = fA[i].y;
                fp fA_z = fA[i].z;

                int j_end = NUMBER_PAR_PER_BOX & ~3;
                for (j = 0; j < j_end; j += 4) {
                    // j
                    r2 = rA_v + rB[j].v - (rA_x * rB[j].x + rA_y * rB[j].y + rA_z * rB[j].z);
                    u2 = a2 * r2;
                    vij = expf(-u2);
                    fs = (fp)2.0 * vij;
                    d.x = rA_x - rB[j].x;
                    d.y = rA_y - rB[j].y;
                    d.z = rA_z - rB[j].z;
                    fxij = fs * d.x;
                    fyij = fs * d.y;
                    fzij = fs * d.z;
                    fA_v += qB[j] * vij;
                    fA_x += qB[j] * fxij;
                    fA_y += qB[j] * fyij;
                    fA_z += qB[j] * fzij;

                    // j+1
                    r2 = rA_v + rB[j + 1].v - (rA_x * rB[j + 1].x + rA_y * rB[j + 1].y + rA_z * rB[j + 1].z);
                    u2 = a2 * r2;
                    vij = expf(-u2);
                    fs = (fp)2.0 * vij;
                    d.x = rA_x - rB[j + 1].x;
                    d.y = rA_y - rB[j + 1].y;
                    d.z = rA_z - rB[j + 1].z;
                    fxij = fs * d.x;
                    fyij = fs * d.y;
                    fzij = fs * d.z;
                    fA_v += qB[j + 1] * vij;
                    fA_x += qB[j + 1] * fxij;
                    fA_y += qB[j + 1] * fyij;
                    fA_z += qB[j + 1] * fzij;

                    // j+2
                    r2 = rA_v + rB[j + 2].v - (rA_x * rB[j + 2].x + rA_y * rB[j + 2].y + rA_z * rB[j + 2].z);
                    u2 = a2 * r2;
                    vij = expf(-u2);
                    fs = (fp)2.0 * vij;
                    d.x = rA_x - rB[j + 2].x;
                    d.y = rA_y - rB[j + 2].y;
                    d.z = rA_z - rB[j + 2].z;
                    fxij = fs * d.x;
                    fyij = fs * d.y;
                    fzij = fs * d.z;
                    fA_v += qB[j + 2] * vij;
                    fA_x += qB[j + 2] * fxij;
                    fA_y += qB[j + 2] * fyij;
                    fA_z += qB[j + 2] * fzij;

                    // j+3
                    r2 = rA_v + rB[j + 3].v - (rA_x * rB[j + 3].x + rA_y * rB[j + 3].y + rA_z * rB[j + 3].z);
                    u2 = a2 * r2;
                    vij = expf(-u2);
                    fs = (fp)2.0 * vij;
                    d.x = rA_x - rB[j + 3].x;
                    d.y = rA_y - rB[j + 3].y;
                    d.z = rA_z - rB[j + 3].z;
                    fxij = fs * d.x;
                    fyij = fs * d.y;
                    fzij = fs * d.z;
                    fA_v += qB[j + 3] * vij;
                    fA_x += qB[j + 3] * fxij;
                    fA_y += qB[j + 3] * fyij;
                    fA_z += qB[j + 3] * fzij;
                }

                for (; j < NUMBER_PAR_PER_BOX; j++) {

                    r2 = rA_v + rB[j].v - (rA_x * rB[j].x + rA_y * rB[j].y + rA_z * rB[j].z);
                    u2 = a2 * r2;
                    vij = expf(-u2);
                    fs = (fp)2.0 * vij;
                    d.x = rA_x - rB[j].x;
                    d.y = rA_y - rB[j].y;
                    d.z = rA_z - rB[j].z;
                    fxij = fs * d.x;
                    fyij = fs * d.y;
                    fzij = fs * d.z;

                    fA_v += qB[j] * vij;
                    fA_x += qB[j] * fxij;
                    fA_y += qB[j] * fyij;
                    fA_z += qB[j] * fzij;
                }

                fA[i].v = fA_v;
                fA[i].x = fA_x;
                fA[i].y = fA_y;
                fA[i].z = fA_z;

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
