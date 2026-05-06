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
    #pragma omp parallel for default(none) \
        shared(knodes, knodes_elem, maxheight, count, currKnode, offset, lastKnode, \
               offset_2, start, end, recstart, reclength, threadsPerBlock)
#endif
    for (int bid = 0; bid < count; bid++) {
        // Private copies for each thread
        long my_currKnode = currKnode[bid];
        long my_offset = offset[bid];
        long my_lastKnode = lastKnode[bid];
        long my_offset_2 = offset_2[bid];
        int my_start = start[bid];
        int my_end = end[bid];
        int my_recstart = recstart[bid];
        int my_reclength = reclength[bid];

        // process levels of the tree
        for (i = 0; i < maxheight; i++) {
            // process all leaves at each level
            for (int thid = 0; thid < threadsPerBlock - 1; thid++) {
                if ((knodes[my_currKnode].keys[thid] <= my_start) &&
                    (knodes[my_currKnode].keys[thid + 1] > my_start)) {
                    if (knodes[my_currKnode].indices[thid] < knodes_elem) {
                        my_offset = knodes[my_currKnode].indices[thid];
                    }
                }
                if ((knodes[my_lastKnode].keys[thid] <= my_end) &&
                    (knodes[my_lastKnode].keys[thid + 1] > my_end)) {
                    if (knodes[my_lastKnode].indices[thid] < knodes_elem) {
                        my_offset_2 = knodes[my_lastKnode].indices[thid];
                    }
                }
            }

            // set for next tree level
            my_currKnode = my_offset;
            my_lastKnode = my_offset_2;
        }

        // process leaves to find starting record
        for (int thid = 0; thid < threadsPerBlock; thid++) {
            if (knodes[my_currKnode].keys[thid] == my_start) {
                my_recstart = knodes[my_currKnode].indices[thid];
            }
        }

        // process leaves to find ending record and calculate length
        for (int thid = 0; thid < threadsPerBlock; thid++) {
            if (knodes[my_lastKnode].keys[thid] == my_end) {
                my_reclength = knodes[my_lastKnode].indices[thid] - my_recstart + 1;
            }
        }

        // Write back results to shared arrays
        currKnode[bid] = my_currKnode;
        offset[bid] = my_offset;
        lastKnode[bid] = my_lastKnode;
        offset_2[bid] = my_offset_2;
        recstart[bid] = my_recstart;
        reclength[bid] = my_reclength;
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
