#include <stdlib.h> // (in directory known to compiler)
#include <stdio.h> // (in directory known to compiler)                  needed by printf

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
    for (bid = 0; bid < count; bid++) {
        long curr_knode_id = currKnode[bid];
        long last_knode_id = lastKnode[bid];
        knode *curr_knode_ptr = &knodes[curr_knode_id];
        knode *last_knode_ptr = &knodes[last_knode_id];
        int start_key = start[bid];
        int end_key = end[bid];

        // process levels of the tree
        for (i = 0; i < maxheight; i++) {
            long new_offset = offset[bid];
            long new_offset_2 = offset_2[bid];

            // process all leaves at each level
            for (thid = 0; thid < threadsPerBlock; thid++) {
                if ((curr_knode_ptr->keys[thid] <= start_key) &&
                    (curr_knode_ptr->keys[thid + 1] > start_key)) {
                    if (curr_knode_ptr->indices[thid] < knodes_elem) {
                        new_offset = curr_knode_ptr->indices[thid];
                    }
                }
                if ((last_knode_ptr->keys[thid] <= end_key) &&
                    (last_knode_ptr->keys[thid + 1] > end_key)) {
                    if (last_knode_ptr->indices[thid] < knodes_elem) {
                        new_offset_2 = last_knode_ptr->indices[thid];
                    }
                }
            }

            // set for next tree level
            curr_knode_id = new_offset;
            last_knode_id = new_offset_2;
            curr_knode_ptr = &knodes[curr_knode_id];
            last_knode_ptr = &knodes[last_knode_id];
            offset[bid] = curr_knode_id;
            offset_2[bid] = last_knode_id;
        }

        currKnode[bid] = curr_knode_id;
        lastKnode[bid] = last_knode_id;

        // process leaves for recstart
        for (thid = 0; thid < threadsPerBlock; thid++) {
            if (curr_knode_ptr->keys[thid] == start_key) {
                recstart[bid] = curr_knode_ptr->indices[thid];
            }
        }

        // process leaves for reclength
        long current_recstart = recstart[bid];
        for (thid = 0; thid < threadsPerBlock; thid++) {
            if (last_knode_ptr->keys[thid] == end_key) {
                reclength[bid] = last_knode_ptr->indices[thid] - current_recstart + 1;
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
