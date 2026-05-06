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
        long off_2 = offset_2[bid];
        int start_key = start[bid];
        int end_key = end[bid];

        // process levels of the tree
        for (i = 0; i < maxheight; i++) {

            knode *curr_node = &knodes[curr];
            knode *last_node = &knodes[last];
            int *curr_keys = curr_node->keys;
            int *last_keys = last_node->keys;
            long *curr_indices = curr_node->indices;
            long *last_indices = last_node->indices;

            // process all leaves at each level
            for (thid = 0; thid < threadsPerBlock; thid++) {

                if ((curr_keys[thid] <= start_key) &&
                    (curr_keys[thid + 1] > start_key)) {

                    long next_off = curr_indices[thid];
                    if (next_off < knodes_elem) {
                        off = next_off;
                    }
                }
                if ((last_keys[thid] <= end_key) &&
                    (last_keys[thid + 1] > end_key)) {

                    long next_off_2 = last_indices[thid];
                    if (next_off_2 < knodes_elem) {
                        off_2 = next_off_2;
                    }
                }
            }

            // set for next tree level
            curr = off;
            last = off_2;
        }

        currKnode[bid] = curr;
        lastKnode[bid] = last;
        offset[bid] = off;
        offset_2[bid] = off_2;

        knode *final_curr_node = &knodes[curr];
        knode *final_last_node = &knodes[last];
        int *final_curr_keys = final_curr_node->keys;
        int *final_last_keys = final_last_node->keys;
        long *final_curr_indices = final_curr_node->indices;
        long *final_last_indices = final_last_node->indices;

        // process leaves
        for (thid = 0; thid < threadsPerBlock; thid++) {

            // Find the index of the starting record
            if (final_curr_keys[thid] == start_key) {
                recstart[bid] = final_curr_indices[thid];
            }
        }

        // process leaves
        for (thid = 0; thid < threadsPerBlock; thid++) {

            // Find the index of the ending record
            if (final_last_keys[thid] == end_key) {
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
