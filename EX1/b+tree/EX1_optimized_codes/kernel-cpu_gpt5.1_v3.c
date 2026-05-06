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

    // process number of queries
    for (int bid = 0; bid < count; bid++) {

        const int key_bid = keys[bid];

        // process levels of the tree
        for (long level = 0; level < maxheight; level++) {

            const long curr = currKnode[bid];
            const knode *curr_node = &knodes[curr];
            long off = offset[bid];

            // process all leaves at each level
            for (int thid = 0; thid < threadsPerBlock; thid++) {

                const int k_this = curr_node->keys[thid];
                const int k_next = curr_node->keys[thid + 1];

                // if value is between the two keys
                if (k_this <= key_bid && k_next > key_bid) {
                    const long idx = knodes[off].indices[thid];
                    if (idx < knodes_elem) {
                        off = idx;
                    }
                }
            }

            // set for next tree level
            offset[bid] = off;
            currKnode[bid] = off;
        }

        const long final_curr = currKnode[bid];
        const knode *final_node = &knodes[final_curr];

        for (int thid = 0; thid < threadsPerBlock; thid++) {

            if (final_node->keys[thid] == key_bid) {
                ans[bid].value = records[final_node->indices[thid]].value;
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
