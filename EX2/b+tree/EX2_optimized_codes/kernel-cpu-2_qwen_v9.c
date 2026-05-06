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
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) private(i)
#endif
    for (int bid = 0; bid < count; bid++) {
        long curr_knode_local = currKnode[bid];
        long offset_local = offset[bid];
        long last_knode_local = lastKnode[bid];
        long offset_2_local = offset_2[bid];
        int start_local = start[bid];
        int end_local = end[bid];

        // process levels of the tree
        for (i = 0; i < maxheight; i++) {

            // process all leaves at each level
            for (int thid = 0; thid < threadsPerBlock - 1; thid++) {

                if ((knodes[curr_knode_local].keys[thid] <= start_local) &&
                    (knodes[curr_knode_local].keys[thid + 1] > start_local)) {

                    if (knodes[curr_knode_local].indices[thid] < knodes_elem) {
                        offset_local = knodes[curr_knode_local].indices[thid];
                    }
                }
                if ((knodes[last_knode_local].keys[thid] <= end_local) &&
                    (knodes[last_knode_local].keys[thid + 1] > end_local)) {

                    if (knodes[last_knode_local].indices[thid] < knodes_elem) {
                        offset_2_local = knodes[last_knode_local].indices[thid];
                    }
                }
            }

            // set for next tree level
            curr_knode_local = offset_local;
            last_knode_local = offset_2_local;
        }

        // Store back the final values after traversal
        currKnode[bid] = curr_knode_local;
        lastKnode[bid] = last_knode_local;
        offset[bid] = offset_local;
        offset_2[bid] = offset_2_local;

        int recstart_local = recstart[bid];

        // process leaves to find starting record
        for (int thid = 0; thid < threadsPerBlock; thid++) {
            if (knodes[curr_knode_local].keys[thid] == start_local) {
                recstart_local = knodes[curr_knode_local].indices[thid];
                break;
            }
        }

        // process leaves to find ending record and calculate length
        for (int thid = 0; thid < threadsPerBlock; thid++) {
            if (knodes[last_knode_local].keys[thid] == end_local) {
                reclength[bid] = knodes[last_knode_local].indices[thid] - recstart_local + 1;
                break;
            }
        }

        // Update the global array with local result
        recstart[bid] = recstart_local;
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
