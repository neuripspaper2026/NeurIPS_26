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
    long i;

// process number of queries
    for (bid = 0; bid < count; ++bid) {

        const int key_bid = keys[bid];
        long curr = currKnode[bid];
        long off = offset[bid];

        // process levels of the tree
        for (i = 0; i < maxheight; ++i) {

            knode *restrict currNode = &knodes[curr];
            knode *restrict offNode  = &knodes[off];

            // process all leaves at each level
            for (thid = 0; thid < threadsPerBlock; ++thid) {

                const int key0 = currNode->keys[thid];
                const int key1 = currNode->keys[thid + 1];

                // if value is between the two keys
                if (key0 <= key_bid && key1 > key_bid) {

                    const long idx = offNode->indices[thid];
                    if (idx < knodes_elem) {
                        off = idx;
                        offNode = &knodes[off];
                    }
                }
            }

            // set for next tree level
            curr = off;
        }

        currKnode[bid] = curr;
        offset[bid] = off;

        knode *restrict leafNode = &knodes[curr];
        record *restrict ans_bid = &ans[bid];

        for (thid = 0; thid < threadsPerBlock; ++thid) {

            if (leafNode->keys[thid] == key_bid) {
                ans_bid->value =
                    records[leafNode->indices[thid]].value;
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
