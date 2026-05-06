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
        long local_currKnode = currKnode[bid];
        long local_lastKnode = lastKnode[bid];
        long local_offset = offset[bid];
        long local_offset_2 = offset_2[bid];
        int local_start = start[bid];
        int local_end = end[bid];
        int local_recstart = recstart[bid];
        int local_reclength = reclength[bid];

        // process levels of the tree
        for (i = 0; i < maxheight; i++) {

            // process all leaves at each level
            for (thid = 0; thid < threadsPerBlock - 1; thid++) { // -1 to prevent out of bounds access

                if ((knodes[local_currKnode].keys[thid] <= local_start) &&
                    (knodes[local_currKnode].keys[thid + 1] > local_start)) {

                    if (knodes[local_currKnode].indices[thid] < knodes_elem) {
                        local_offset = knodes[local_currKnode].indices[thid];
                    }
                }
                if ((knodes[local_lastKnode].keys[thid] <= local_end) &&
                    (knodes[local_lastKnode].keys[thid + 1] > local_end)) {

                    if (knodes[local_lastKnode].indices[thid] < knodes_elem) {
                        local_offset_2 = knodes[local_lastKnode].indices[thid];
                    }
                }
            }

            // set for next tree level
            local_currKnode = local_offset;
            local_lastKnode = local_offset_2;
        }

        // process leaves to find starting record
        for (thid = 0; thid < threadsPerBlock; thid++) {
            // Find the index of the starting record
            if (knodes[local_currKnode].keys[thid] == local_start) {
                local_recstart = knodes[local_currKnode].indices[thid];
            }
        }

        // process leaves to find ending record
        for (thid = 0; thid < threadsPerBlock; thid++) {
            // Find the index of the ending record
            if (knodes[local_lastKnode].keys[thid] == local_end) {
                local_reclength =
                    knodes[local_lastKnode].indices[thid] - local_recstart + 1;
            }
        }

        // Update the arrays with local values
        currKnode[bid] = local_currKnode;
        lastKnode[bid] = local_lastKnode;
        offset[bid] = local_offset;
        offset_2[bid] = local_offset_2;
        recstart[bid] = local_recstart;
        reclength[bid] = local_reclength;
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
