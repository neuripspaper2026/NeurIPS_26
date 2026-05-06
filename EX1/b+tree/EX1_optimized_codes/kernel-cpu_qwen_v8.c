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

    // Cache frequently accessed pointers
    knode *current_knode_ptr;
    knode *offset_knode_ptr;
    long current_knode_idx;
    long offset_idx;

// process number of queries
    for (bid = 0; bid < count; bid++) {
        current_knode_idx = currKnode[bid];
        offset_idx = offset[bid];
        current_knode_ptr = &knodes[current_knode_idx];

        // process levels of the tree
        for (i = 0; i < maxheight; i++) {

            // process all leaves at each level
            for (thid = 0; thid < threadsPerBlock - 1; thid++) { // -1 because we access thid+1

                // if value is between the two keys
                if ((current_knode_ptr->keys[thid]) <= keys[bid] &&
                    (current_knode_ptr->keys[thid + 1] > keys[bid])) {

                    if (current_knode_ptr->indices[thid] < knodes_elem) {
                        offset_idx = current_knode_ptr->indices[thid];
                        break; // Found the matching index, no need to continue
                    }
                }
            }

            // set for next tree level
            current_knode_idx = offset_idx;
            current_knode_ptr = &knodes[current_knode_idx];
        }

        offset[bid] = offset_idx;
        currKnode[bid] = current_knode_idx;

        // Final search for exact key match
        for (thid = 0; thid < threadsPerBlock; thid++) {
            if (current_knode_ptr->keys[thid] == keys[bid]) {
                ans[bid].value = records[current_knode_ptr->indices[thid]].value;
                break; // Found the key, exit loop
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
