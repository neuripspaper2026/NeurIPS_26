#include <stdlib.h> // (in directory known to compiler)
#include <stdio.h>  // (in directory known to compiler)                  needed by printf

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

    int threadsPerBlock = order < 1024 ? order : 1024;

    time1 = get_time();

    // process number of queries
    for (int bid = 0; bid < count; ++bid) {

        // process levels of the tree
        for (long level = 0; level < maxheight; ++level) {

            const long curr_idx = currKnode[bid];
            const long last_idx = lastKnode[bid];

            const knode *curr_node = &knodes[curr_idx];
            const knode *last_node = &knodes[last_idx];

            const int start_key = start[bid];
            const int end_key = end[bid];

            long next_curr_offset = offset[bid];
            long next_last_offset = offset_2[bid];

            // process all leaves at each level
            for (int thid = 0; thid < threadsPerBlock; ++thid) {

                const int curr_key_low = curr_node->keys[thid];
                const int curr_key_high = curr_node->keys[thid + 1];

                if (curr_key_low <= start_key && curr_key_high > start_key) {

                    const long child_index = curr_node->indices[thid];
                    if (child_index < knodes_elem) {
                        next_curr_offset = child_index;
                    }
                }

                const int last_key_low = last_node->keys[thid];
                const int last_key_high = last_node->keys[thid + 1];

                if (last_key_low <= end_key && last_key_high > end_key) {

                    const long child_index2 = last_node->indices[thid];
                    if (child_index2 < knodes_elem) {
                        next_last_offset = child_index2;
                    }
                }
            }

            // set for next tree level
            currKnode[bid] = next_curr_offset;
            lastKnode[bid] = next_last_offset;
            offset[bid] = next_curr_offset;
            offset_2[bid] = next_last_offset;
        }

        const long leaf_start_idx = currKnode[bid];
        const long leaf_end_idx = lastKnode[bid];

        const knode *leaf_start_node = &knodes[leaf_start_idx];
        const knode *leaf_end_node = &knodes[leaf_end_idx];

        const int start_key = start[bid];
        const int end_key = end[bid];

        // process leaves: find index of the starting record
        for (int thid = 0; thid < threadsPerBlock; ++thid) {

            if (leaf_start_node->keys[thid] == start_key) {
                recstart[bid] = leaf_start_node->indices[thid];
            }
        }

        // process leaves: find index of the ending record
        for (int thid = 0; thid < threadsPerBlock; ++thid) {

            if (leaf_end_node->keys[thid] == end_key) {
                reclength[bid] =
                    leaf_end_node->indices[thid] - recstart[bid] + 1;
            }
        }
    }

    time2 = get_time();

    printf("Time spent in different stages of CPU/MCPU KERNEL:\n");

    printf("%15.12f s, %15.12f % : MCPU: SET DEVICE\n",
           (float)(time1 - time0) / 1000000.0f,
           (float)(time1 - time0) / (float)(time2 - time0) * 100.0f);
    printf("%15.12f s, %15.12f % : CPU/MCPU: KERNEL\n",
           (float)(time2 - time1) / 1000000.0f,
           (float)(time2 - time1) / (float)(time2 - time0) * 100.0f);

    printf("Total time:\n");
    printf("%.12f s\n", (float)(time2 - time0) / 1000000.0f);

}
