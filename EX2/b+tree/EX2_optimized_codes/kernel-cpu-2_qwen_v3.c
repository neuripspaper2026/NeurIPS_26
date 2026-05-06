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

// process number of queries
#pragma omp parallel for schedule(dynamic)
    for (int bid = 0; bid < count; bid++) {
        // private thread IDs
        int thid;

        long currKnode_local = currKnode[bid];
        long lastKnode_local = lastKnode[bid];
        long offset_local = offset[bid];
        long offset_2_local = offset_2[bid];
        int start_local = start[bid];
        int end_local = end[bid];
        int recstart_local = recstart[bid];
        int reclength_local = reclength[bid];

        // process levels of the tree
        for (i = 0; i < maxheight; i++) {

            // process all leaves at each level
            for (thid = 0; thid < threadsPerBlock - 1; thid++) {

                if ((knodes[currKnode_local].keys[thid] <= start_local) &&
                    (knodes[currKnode_local].keys[thid + 1] > start_local)) {

                    if (knodes[currKnode_local].indices[thid] < knodes_elem) {
                        offset_local = knodes[currKnode_local].indices[thid];
                    }
                }
                if ((knodes[lastKnode_local].keys[thid] <= end_local) &&
                    (knodes[lastKnode_local].keys[thid + 1] > end_local)) {

                    if (knodes[lastKnode_local].indices[thid] < knodes_elem) {
                        offset_2_local = knodes[lastKnode_local].indices[thid];
                    }
                }
            }

            // set for next tree level
            currKnode_local = offset_local;
            lastKnode_local = offset_2_local;
        }

        // process leaves
        for (thid = 0; thid < threadsPerBlock; thid++) {

            // Find the index of the starting record
            if (knodes[currKnode_local].keys[thid] == start_local) {
                recstart_local = knodes[currKnode_local].indices[thid];
            }
        }

        // process leaves
        for (thid = 0; thid < threadsPerBlock; thid++) {

            // Find the index of the ending record
            if (knodes[lastKnode_local].keys[thid] == end_local) {
                reclength_local =
                    knodes[lastKnode_local].indices[thid] - recstart_local + 1;
            }
        }

        // Update global arrays with local values
        currKnode[bid] = currKnode_local;
        lastKnode[bid] = lastKnode_local;
        offset[bid] = offset_local;
        offset_2[bid] = offset_2_local;
        recstart[bid] = recstart_local;
        reclength[bid] = reclength_local;
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
