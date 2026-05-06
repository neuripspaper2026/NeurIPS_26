#include <stdlib.h> // (in directory known to compiler)			needed by malloc
#include <stdio.h> // (in directory known to compiler)			needed by printf, stderr

#include "../common.h" // (in directory provided here)

#include "../util/timer/timer.h" // (in directory provided here)
#include "../kernel_cpu.h" // (in directory provided here)

void kernel_cpu(record *records, knode *knodes, long knodes_elem,

                int order, long maxheight, int count,

                long *currKnode, long *offset, int *keys, record *ans) {

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

        const int key_bid = keys[bid];

        // process levels of the tree
        for (level = 0; level < maxheight; level++) {

            const long curr_idx = currKnode[bid];
            const knode *curr_node = &knodes[curr_idx];
            const int *restrict node_keys = curr_node->keys;

            const long off_idx = offset[bid];
            const knode *off_node = &knodes[off_idx];
            const long *restrict node_indices = off_node->indices;

            // process all leaves at each level
            for (thid = 0; thid < threadsPerBlock; thid++) {

                const int key_left = node_keys[thid];
                const int key_right = node_keys[thid + 1];

                // if value is between the two keys
                if (key_left <= key_bid && key_right > key_bid) {
                    const long child_index = node_indices[thid];
                    if (child_index < knodes_elem) {
                        offset[bid] = child_index;
                    }
                }
            }

            // set for next tree level
            currKnode[bid] = offset[bid];
        }

        {
            const long leaf_idx = currKnode[bid];
            const knode *leaf_node = &knodes[leaf_idx];
            const int *restrict leaf_keys = leaf_node->keys;
            const long *restrict leaf_indices = leaf_node->indices;

            for (thid = 0; thid < threadsPerBlock; thid++) {
                if (leaf_keys[thid] == key_bid) {
                    const long rec_index = leaf_indices[thid];
                    ans[bid].value = records[rec_index].value;
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
