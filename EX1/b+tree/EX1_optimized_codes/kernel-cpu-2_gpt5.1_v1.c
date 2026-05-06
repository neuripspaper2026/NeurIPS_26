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

    int threadsPerBlock = (order < 1024) ? order : 1024;

    time1 = get_time();

    // process number of queries
    for (int bid = 0; bid < count; ++bid) {

        const int start_key = start[bid];
        const int end_key = end[bid];

        // process levels of the tree
        for (long level = 0; level < maxheight; ++level) {

            const long curr_idx = currKnode[bid];
            const long last_idx = lastKnode[bid];

            const knode *curr_node = &knodes[curr_idx];
            const knode *last_node = &knodes[last_idx];

            // process all leaves at each level
            for (int thid = 0; thid < threadsPerBlock; ++thid) {

                const int c_k0 = curr_node->keys[thid];
                const int c_k1 = curr_node->keys[thid + 1];

                if (c_k0 <= start_key && c_k1 > start_key) {
                    const long next_idx = curr_node->indices[thid];
                    if (next_idx < knodes_elem) {
                        offset[bid] = next_idx;
                    }
                }

                const int l_k0 = last_node->keys[thid];
                const int l_k1 = last_node->keys[thid + 1];

                if (l_k0 <= end_key && l_k1 > end_key) {
                    const long next_idx2 = last_node->indices[thid];
                    if (next_idx2 < knodes_elem) {
                        offset_2[bid] = next_idx2;
                    }
                }
            }

            // set for next tree level
            currKnode[bid] = offset[bid];
            lastKnode[bid] = offset_2[bid];
        }

        const long leaf_start_idx = currKnode[bid];
        const long leaf_end_idx = lastKnode[bid];

        const knode *leaf_start_node = &knodes[leaf_start_idx];
        const knode *leaf_end_node = &knodes[leaf_end_idx];

        // process leaves: find the index of the starting record
        for (int thid = 0; thid < threadsPerBlock; ++thid) {

            if (leaf_start_node->keys[thid] == start_key) {
                recstart[bid] = leaf_start_node->indices[thid];
            }
        }

        // process leaves: find the index of the ending record
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
           (float)(time1 - time0) / 1000000,
           (float)(time1 - time0) / (float)(time2 - time0) * 100);
    printf("%15.12f s, %15.12f % : CPU/MCPU: KERNEL\n",
           (float)(time2 - time1) / 1000000,
           (float)(time2 - time1) / (float)(time2 - time0) * 100);

    printf("Total time:\n");
    printf("%.12f s\n", (float)(time2 - time0) / 1000000);
}
