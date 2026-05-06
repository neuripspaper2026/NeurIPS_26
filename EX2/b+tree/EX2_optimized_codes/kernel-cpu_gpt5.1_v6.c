#include <stdlib.h> // (in directory known to compiler)			needed by malloc
#include <stdio.h> // (in directory known to compiler)			needed by printf, stderr
#ifdef _OPENMP
#include <omp.h>
#endif

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
#ifdef _OPENMP
#pragma omp parallel for default(none) private(bid, i, thid) shared(count, maxheight, threadsPerBlock, knodes, knodes_elem, currKnode, offset, keys, records, ans)
#endif
    for (bid = 0; bid < count; bid++) {

        long localCurrKnode = currKnode[bid];
        long localOffset = offset[bid];
        const int key = keys[bid];

        // process levels of the tree
        for (i = 0; i < maxheight; i++) {

            knode *currNode = &knodes[localCurrKnode];
            knode *offNode  = &knodes[localOffset];

            // process all leaves at each level
            for (thid = 0; thid < threadsPerBlock; thid++) {

                int keyL = currNode->keys[thid];
                int keyR = currNode->keys[thid + 1];

                // if value is between the two keys
                if (keyL <= key && keyR > key) {

                    long idx = offNode->indices[thid];
                    if (idx < knodes_elem) {
                        localOffset = idx;
                        offNode = &knodes[localOffset];
                    }
                }
            }

            // set for next tree level
            localCurrKnode = localOffset;
        }

        knode *finalNode = &knodes[localCurrKnode];

        for (thid = 0; thid < threadsPerBlock; thid++) {

            if (finalNode->keys[thid] == key) {
                ans[bid].value = records[finalNode->indices[thid]].value;
            }
        }

        currKnode[bid] = localCurrKnode;
        offset[bid] = localOffset;
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
