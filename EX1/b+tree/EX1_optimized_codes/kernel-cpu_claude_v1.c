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
    int i;

// process number of queries
    for (bid = 0; bid < count; bid++) {

        long curr = currKnode[bid];
        long off = offset[bid];
        int key = keys[bid];

        // process levels of the tree
        for (i = 0; i < maxheight; i++) {

            knode *curr_node = &knodes[curr];
            int *curr_keys = curr_node->keys;
            long *off_indices = knodes[off].indices;

            // process all leaves at each level
            for (thid = 0; thid < threadsPerBlock; thid++) {

                // if value is between the two keys
                if (curr_keys[thid] <= key && curr_keys[thid + 1] > key) {

                    long next_off = off_indices[thid];
                    if (next_off < knodes_elem) {
                        off = next_off;
                    }
                }
            }

            // set for next tree level
            curr = off;
        }

        currKnode[bid] = curr;

        knode *final_node = &knodes[curr];
        int *final_keys = final_node->keys;
        long *final_indices = final_node->indices;

        for (thid = 0; thid < threadsPerBlock; thid++) {

            if (final_keys[thid] == key) {
                ans[bid].value = records[final_indices[thid]].value;
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
