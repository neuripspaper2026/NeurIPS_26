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

    time0 = get_time();

    int threadsPerBlock;
    threadsPerBlock = order < 1024 ? order : 1024;

    time1 = get_time();

    // private thread IDs
    int thid;
    int bid;
    long level;

// process number of queries
    for (bid = 0; bid < count; bid++) {

        const int start_bid = start[bid];
        const int end_bid = end[bid];

        // process levels of the tree
        for (level = 0; level < maxheight; level++) {

            const long curr_idx = currKnode[bid];
            const long last_idx = lastKnode[bid];

            const knode *curr_node = &knodes[curr_idx];
            const knode *last_node = &knodes[last_idx];

            const int *restrict curr_keys = curr_node->keys;
            const long *restrict curr_indices = curr_node->indices;

            const int *restrict last_keys = last_node->keys;
            const long *restrict last_indices = last_node->indices;

            // process all leaves at each level
            for (thid = 0; thid < threadsPerBlock; thid++) {

                const int curr_key_left = curr_keys[thid];
                const int curr_key_right = curr_keys[thid + 1];

                if (curr_key_left <= start_bid && curr_key_right > start_bid) {
                    const long child_index = curr_indices[thid];
                    if (child_index < knodes_elem) {
                        offset[bid] = child_index;
                    }
                }

                const int last_key_left = last_keys[thid];
                const int last_key_right = last_keys[thid + 1];

                if (last_key_left <= end_bid && last_key_right > end_bid) {
                    const long child_index2 = last_indices[thid];
                    if (child_index2 < knodes_elem) {
                        offset_2[bid] = child_index2;
                    }
                }
            }

            // set for next tree level
            currKnode[bid] = offset[bid];
            lastKnode[bid] = offset_2[bid];
        }

        // process leaves: Find the index of the starting record
        {
            const long leaf_start_idx = currKnode[bid];
            const knode *leaf_start_node = &knodes[leaf_start_idx];
            const int *restrict leaf_start_keys = leaf_start_node->keys;
            const long *restrict leaf_start_indices = leaf_start_node->indices;

            for (thid = 0; thid < threadsPerBlock; thid++) {
                if (leaf_start_keys[thid] == start_bid) {
                    recstart[bid] = leaf_start_indices[thid];
                }
            }
        }

        // process leaves: Find the index and length of the ending record
        {
            const long leaf_end_idx = lastKnode[bid];
            const knode *leaf_end_node = &knodes[leaf_end_idx];
            const int *restrict leaf_end_keys = leaf_end_node->keys;
            const long *restrict leaf_end_indices = leaf_end_node->indices;

            for (thid = 0; thid < threadsPerBlock; thid++) {
                if (leaf_end_keys[thid] == end_bid) {
                    reclength[bid] =
                        leaf_end_indices[thid] - recstart[bid] + 1;
                }
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
