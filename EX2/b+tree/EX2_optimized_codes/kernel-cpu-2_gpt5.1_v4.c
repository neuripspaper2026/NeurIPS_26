#include <stdlib.h> // (in directory known to compiler)
#include <stdio.h>  // (in directory known to compiler)                  needed by printf
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

    // process number of queries
#ifdef _OPENMP
#pragma omp parallel for default(none) schedule(static) \
    shared(count, maxheight, threadsPerBlock, knodes, knodes_elem,       \
           currKnode, offset, lastKnode, offset_2, start, end, recstart, \
           reclength)                                                    \
    if(count > 1)
#endif
    for (int bid = 0; bid < count; bid++) {

        long curr     = currKnode[bid];
        long last     = lastKnode[bid];
        long off      = offset[bid];
        long off2     = offset_2[bid];
        const int s   = start[bid];
        const int e   = end[bid];
        const int tpb = threadsPerBlock;

        // process levels of the tree
        for (i = 0; i < maxheight; i++) {

            // process all leaves at each level
            for (int thid = 0; thid < tpb; thid++) {

                const int k_curr     = knodes[curr].keys[thid];
                const int k_curr_nxt = knodes[curr].keys[thid + 1];

                if (k_curr <= s && k_curr_nxt > s) {
                    const long idx = knodes[curr].indices[thid];
                    if (idx < knodes_elem) {
                        off = idx;
                    }
                }

                const int k_last     = knodes[last].keys[thid];
                const int k_last_nxt = knodes[last].keys[thid + 1];

                if (k_last <= e && k_last_nxt > e) {
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

        currKnode[bid] = curr;
        lastKnode[bid] = last;
        offset[bid]    = off;
        offset_2[bid]  = off2;

        // process leaves - Find the index of the starting record
        for (int thid = 0; thid < tpb; thid++) {

            if (knodes[curr].keys[thid] == s) {
                recstart[bid] = knodes[curr].indices[thid];
            }
        }

        // process leaves - Find the index of the ending record
        for (int thid = 0; thid < tpb; thid++) {

            if (knodes[last].keys[thid] == e) {
                reclength[bid] = knodes[last].indices[thid] - recstart[bid] + 1;
            }
        }
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
