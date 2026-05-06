#include <stdlib.h>
#include <stdio.h>

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
    int thid;
    int bid;

// process number of queries
    for (bid = 0; bid < count; bid++) {

        long curr = currKnode[bid];
        long last = lastKnode[bid];
        long off = offset[bid];
        long off2 = offset_2[bid];
        int start_val = start[bid];
        int end_val = end[bid];

        // process levels of the tree
        for (i = 0; i < maxheight; i++) {

            knode *curr_knode = &knodes[curr];
            knode *last_knode = &knodes[last];
            int *curr_keys = curr_knode->keys;
            int *last_keys = last_knode->keys;
            long *curr_indices = curr_knode->indices;
            long *last_indices = last_knode->indices;

            // process all leaves at each level
            for (thid = 0; thid < threadsPerBlock; thid++) {

                if ((curr_keys[thid] <= start_val) &&
                    (curr_keys[thid + 1] > start_val)) {

                    long idx = curr_indices[thid];
                    if (idx < knodes_elem) {
                        off = idx;
                    }
                }
                if ((last_keys[thid] <= end_val) &&
                    (last_keys[thid + 1] > end_val)) {

                    long idx = last_indices[thid];
                    if (idx < knodes_elem) {
                        off2 = idx;
                    }
                }
            }

            // set for next tree level
            curr = off;
            last = off2;
        }

        currKnode[bid] = curr;
        lastKnode[bid] = last;
        offset[bid] = off;
        offset_2[bid] = off2;

        knode *final_curr_knode = &knodes[curr];
        knode *final_last_knode = &knodes[last];
        int *final_curr_keys = final_curr_knode->keys;
        int *final_last_keys = final_last_knode->keys;
        long *final_curr_indices = final_curr_knode->indices;
        long *final_last_indices = final_last_knode->indices;

        // process leaves
        for (thid = 0; thid < threadsPerBlock; thid++) {

            // Find the index of the starting record
            if (final_curr_keys[thid] == start_val) {
                recstart[bid] = final_curr_indices[thid];
            }
        }

        // process leaves
        for (thid = 0; thid < threadsPerBlock; thid++) {

            // Find the index of the ending record
            if (final_last_keys[thid] == end_val) {
                reclength[bid] =
                    final_last_indices[thid] - recstart[bid] + 1;
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
