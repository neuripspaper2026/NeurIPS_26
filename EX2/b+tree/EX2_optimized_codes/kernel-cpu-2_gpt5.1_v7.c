#include <stdlib.h> // (in directory known to compiler)
#include <stdio.h> // (in directory known to compiler)                  needed by printf
#ifdef _OPENMP
#include <omp.h>
#endif

#include "../common.h" // (in directory provided here)

#include "../util/timer/timer.h" // (in directory provided here)	needed by timer

#include "../kernel_cpu_2.h" // (in directory provided here)

void kernel_cpu_2(knode *knodes, long knodes_elem,

                  int order, long maxheight, int count,

                  long *currKnode, long *offset, long *lastKnode,
                  long *offset_2, int *start, int *end, int *recstart,
                  int *reclength) {

    // timer
    long long time0;
    long long time1;
    long long time2;

    // common variables
    int i;

    time0 = get_time();

    int threadsPerBlock;
    threadsPerBlock = order < 1024 ? order : 1024;

    time1 = get_time();

    // private thread IDs
    int thid;
    int bid;

    // process number of queries
#ifdef _OPENMP
#pragma omp parallel for default(none) private(i, thid) shared(knodes, knodes_elem, order, maxheight, count, currKnode, offset, lastKnode, offset_2, start, end, recstart, reclength, threadsPerBlock) schedule(static)
#endif
    for (bid = 0; bid < count; bid++) {

        long curr  = currKnode[bid];
        long last  = lastKnode[bid];
        long off   = offset[bid];
        long off2  = offset_2[bid];
        const int s = start[bid];
        const int e = end[bid];

        // process levels of the tree
        for (i = 0; i < maxheight; i++) {

            // process all leaves at each level
#ifdef _OPENMP
#pragma omp simd
#endif
            for (thid = 0; thid < threadsPerBlock; thid++) {

                const int k_curr_left  = knodes[curr].keys[thid];
                const int k_curr_right = knodes[curr].keys[thid + 1];

                if (k_curr_left <= s && k_curr_right > s) {
                    const long idx = knodes[curr].indices[thid];
                    if (idx < knodes_elem) {
                        off = idx;
                    }
                }

                const int k_last_left  = knodes[last].keys[thid];
                const int k_last_right = knodes[last].keys[thid + 1];

                if (k_last_left <= e && k_last_right > e) {
                    const long idx2 = knodes[last].indices[thid];
                    if (idx2 < knodes_elem) {
                        off2 = idx2;
                    }
                }
            }

            // set for next tree level
            curr = off;
            last = off2;
        }

        // process leaves - find the index of the starting record
#ifdef _OPENMP
#pragma omp simd
#endif
        for (thid = 0; thid < threadsPerBlock; thid++) {
            if (knodes[curr].keys[thid] == s) {
                recstart[bid] = knodes[curr].indices[thid];
            }
        }

        // process leaves - find the index and length of the ending record
#ifdef _OPENMP
#pragma omp simd
#endif
        for (thid = 0; thid < threadsPerBlock; thid++) {
            if (knodes[last].keys[thid] == e) {
                reclength[bid] = knodes[last].indices[thid] - recstart[bid] + 1;
            }
        }

        currKnode[bid] = curr;
        lastKnode[bid] = last;
        offset[bid] = off;
        offset_2[bid] = off2;
    }

    time2 = get_time();

    printf("Time spent in different stages of CPU/MCPU KERNEL:\n");

    printf("%15.12f s, %15.12f % : MCPU: SET DEVICE\n",
           (float)(time1 - time0) / 1000000,
           (float)(time1 - time0) / (float)(time2 - time0) * 100);
    printf("%15.12f s, %15.12f % : CPU/MCPU: KERNEL\n",
           (float)(time2 - time1) / 1000000,
           (float)(time2 - time1) / (float)(time2 - time0) * 100);

    printf("Total time:\n");
    printf("%.12f s\n", (float)(time2 - time0) / 1000000);

}
