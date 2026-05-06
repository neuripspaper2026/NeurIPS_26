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
    #pragma omp parallel for schedule(dynamic) private(i)
    for (int bid = 0; bid < count; bid++) {

        // process levels of the tree
        for (i = 0; i < maxheight; i++) {

            long curr_knode_val = currKnode[bid];
            long last_knode_val = lastKnode[bid];
            long offset_val = offset[bid];
            long offset_2_val = offset_2[bid];

            // process all leaves at each level
            #pragma omp simd
            for (int thid = 0; thid < threadsPerBlock - 1; thid++) {

                if ((knodes[curr_knode_val].keys[thid] <= start[bid]) &&
                    (knodes[curr_knode_val].keys[thid + 1] > start[bid])) {

                    if (knodes[curr_knode_val].indices[thid] < knodes_elem) {
                        offset_val = knodes[curr_knode_val].indices[thid];
                    }
                }
                if ((knodes[last_knode_val].keys[thid] <= end[bid]) &&
                    (knodes[last_knode_val].keys[thid + 1] > end[bid])) {

                    if (knodes[last_knode_val].indices[thid] < knodes_elem) {
                        offset_2_val = knodes[last_knode_val].indices[thid];
                    }
                }
            }

            // set for next tree level
            currKnode[bid] = offset_val;
            lastKnode[bid] = offset_2_val;
            offset[bid] = offset_val;
            offset_2[bid] = offset_2_val;
        }

        long curr_knode_final = currKnode[bid];
        long last_knode_final = lastKnode[bid];

        // process leaves to find starting record
        #pragma omp simd
        for (int thid = 0; thid < threadsPerBlock; thid++) {
            // Find the index of the starting record
            if (knodes[curr_knode_final].keys[thid] == start[bid]) {
                recstart[bid] = knodes[curr_knode_final].indices[thid];
            }
        }

        // process leaves to find ending record
        #pragma omp simd
        for (int thid = 0; thid < threadsPerBlock; thid++) {
            // Find the index of the ending record
            if (knodes[last_knode_final].keys[thid] == end[bid]) {
                reclength[bid] =
                    knodes[last_knode_final].indices[thid] - recstart[bid] + 1;
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
