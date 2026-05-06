#include <stdlib.h> // (in directory known to compiler)
#include <stdio.h>  // (in directory known to compiler)                  needed by printf

#include "../common.h" // (in directory provided here)

#include "../util/timer/timer.h" // (in directory provided here)  needed by timer

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
    long i;

// process number of queries
    for (bid = 0; bid < count; ++bid) {

        const int start_bid = start[bid];
        const int end_bid   = end[bid];

        long curr  = currKnode[bid];
        long last  = lastKnode[bid];
        long off   = offset[bid];
        long off_2 = offset_2[bid];

        // process levels of the tree
        for (i = 0; i < maxheight; ++i) {

            knode *restrict currNode = &knodes[curr];
            knode *restrict lastNode = &knodes[last];

            // process all leaves at each level
            for (thid = 0; thid < threadsPerBlock; ++thid) {

                const int ckey0 = currNode->keys[thid];
                const int ckey1 = currNode->keys[thid + 1];

                if (ckey0 <= start_bid && ckey1 > start_bid) {

                    const long idx = currNode->indices[thid];
                    if (idx < knodes_elem) {
                        off = idx;
                    }
                }

                const int lkey0 = lastNode->keys[thid];
                const int lkey1 = lastNode->keys[thid + 1];

                if (lkey0 <= end_bid && lkey1 > end_bid) {

                    const long idx2 = lastNode->indices[thid];
                    if (idx2 < knodes_elem) {
                        off_2 = idx2;
                    }
                }
            }

            // set for next tree level
            curr = off;
            last = off_2;
        }

        currKnode[bid] = curr;
        lastKnode[bid] = last;
        offset[bid]    = off;
        offset_2[bid]  = off_2;

        knode *restrict leafStart = &knodes[curr];
        knode *restrict leafEnd   = &knodes[last];

        // process leaves: Find the index of the starting record
        for (thid = 0; thid < threadsPerBlock; ++thid) {

            if (leafStart->keys[thid] == start_bid) {
                recstart[bid] = leafStart->indices[thid];
            }
        }

        // process leaves: Find the index of the ending record
        for (thid = 0; thid < threadsPerBlock; ++thid) {

            if (leafEnd->keys[thid] == end_bid) {
                reclength[bid] =
                    leafEnd->indices[thid] - recstart[bid] + 1;
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
