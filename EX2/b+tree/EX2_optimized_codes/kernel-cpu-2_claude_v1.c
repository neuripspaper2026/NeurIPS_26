#include <stdlib.h>
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "../common.h"

#include "../util/timer/timer.h"

#include "../kernel_cpu_2.h"

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
    int bid;

// process number of queries
#ifdef _OPENMP
    #pragma omp parallel for private(bid, i) schedule(dynamic, 1)
#endif
    for (bid = 0; bid < count; bid++) {
        long curr = currKnode[bid];
        long last = lastKnode[bid];
        long off = offset[bid];
        long off2 = offset_2[bid];
        int start_val = start[bid];
        int end_val = end[bid];

        // process levels of the tree
        for (i = 0; i < maxheight; i++) {
            int thid;
            int found_start = 0;
            int found_end = 0;
            long next_off = off;
            long next_off2 = off2;

            // process all leaves at each level
            for (thid = 0; thid < threadsPerBlock; thid++) {
                if (!found_start &&
                    (knodes[curr].keys[thid] <= start_val) &&
                    (knodes[curr].keys[thid + 1] > start_val)) {

                    long idx = knodes[curr].indices[thid];
                    if (idx < knodes_elem) {
                        next_off = idx;
                        found_start = 1;
                    }
                }
                if (!found_end &&
                    (knodes[last].keys[thid] <= end_val) &&
                    (knodes[last].keys[thid + 1] > end_val)) {

                    long idx = knodes[last].indices[thid];
                    if (idx < knodes_elem) {
                        next_off2 = idx;
                        found_end = 1;
                    }
                }
                if (found_start && found_end) {
                    break;
                }
            }

            if (found_start) {
                off = next_off;
            }
            if (found_end) {
                off2 = next_off2;
            }

            // set for next tree level
            curr = off;
            last = off2;
        }

        // process leaves - find starting record
        int thid;
        for (thid = 0; thid < threadsPerBlock; thid++) {
            if (knodes[curr].keys[thid] == start_val) {
                recstart[bid] = knodes[curr].indices[thid];
                break;
            }
        }

        // process leaves - find ending record
        for (thid = 0; thid < threadsPerBlock; thid++) {
            if (knodes[last].keys[thid] == end_val) {
                reclength[bid] =
                    knodes[last].indices[thid] - recstart[bid] + 1;
                break;
            }
        }

        // Update global arrays
        offset[bid] = off;
        offset_2[bid] = off2;
        currKnode[bid] = curr;
        lastKnode[bid] = last;
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
