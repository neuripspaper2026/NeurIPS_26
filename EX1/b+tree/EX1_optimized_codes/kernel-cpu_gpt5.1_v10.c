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
    for (bid = 0; bid < count; ++bid) {

        const int key = keys[bid];

        // process levels of the tree
        for (level = 0; level < maxheight; ++level) {

            const long curr_idx = currKnode[bid];
            const long off_idx  = offset[bid];

            knode *const curr_node = &knodes[curr_idx];
            knode *const off_node  = &knodes[off_idx];

            int selected_thid = -1;

            // process all leaves at each level
            for (thid = 0; thid < threadsPerBlock; ++thid) {

                const int key_left  = curr_node->keys[thid];
                const int key_right = curr_node->keys[thid + 1];

                // if value is between the two keys
                if (key_left <= key && key_right > key) {
                    const long idx = off_node->indices[thid];
                    if (idx < knodes_elem) {
                        offset[bid] = idx;
                    }
                    selected_thid = thid;
                    break;
                }
            }

            // set for next tree level
            currKnode[bid] = offset[bid];
        }

        const long leaf_idx = currKnode[bid];
        knode *const leaf_node = &knodes[leaf_idx];

        for (thid = 0; thid < threadsPerBlock; ++thid) {

            if (leaf_node->keys[thid] == key) {
                ans[bid].value = records[leaf_node->indices[thid]].value;
                break;
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
