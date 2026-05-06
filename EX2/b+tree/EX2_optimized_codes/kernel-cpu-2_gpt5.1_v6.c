#include <stdlib.h> // (in directory known to compiler)
#include <stdio.h>  // (in directory known to compiler)                  needed by printf
#ifdef _OPENMP
#include <omp.h>
#endif

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

    // common variables
    int i;

    time0 = get_time();

    int threadsPerBlock;
    threadsPerBlock = order < 1024 ? order : 1024;

    time1 = get_time();

    // private thread IDs
    int thid;
    int bid;

    // process number of queries
#ifdef _OPENMP
#pragma omp parallel for default(none) private(bid, i, thid) shared(count, maxheight, threadsPerBlock, knodes, knodes_elem, currKnode, offset, lastKnode, offset_2, start, end, recstart, reclength)
#endif
    for (bid = 0; bid < count; bid++) {

        long localCurrKnode = currKnode[bid];
        long localLastKnode = lastKnode[bid];
        long localOffset    = offset[bid];
        long localOffset_2  = offset_2[bid];
        const int startKey  = start[bid];
        const int endKey    = end[bid];

        // process levels of the tree
        for (i = 0; i < maxheight; i++) {

            knode *currNode = &knodes[localCurrKnode];
            knode *lastNode = &knodes[localLastKnode];

            // process all leaves at each level
            for (thid = 0; thid < threadsPerBlock; thid++) {

                int keyL  = currNode->keys[thid];
                int keyR  = currNode->keys[thid + 1];
                int lkeyL = lastNode->keys[thid];
                int lkeyR = lastNode->keys[thid + 1];

                if (keyL <= startKey && keyR > startKey) {
                    long idx = currNode->indices[thid];
                    if (idx < knodes_elem) {
                        localOffset = idx;
                    }
                }

                if (lkeyL <= endKey && lkeyR > endKey) {
                    long idx2 = lastNode->indices[thid];
                    if (idx2 < knodes_elem) {
                        localOffset_2 = idx2;
                    }
                }
            }

            // set for next tree level
            localCurrKnode = localOffset;
            localLastKnode = localOffset_2;
        }

        // process leaves: find index of the starting record
        {
            knode *currNode = &knodes[localCurrKnode];
            int   rs        = recstart[bid]; // in case not found, keep previous
            for (thid = 0; thid < threadsPerBlock; thid++) {
                if (currNode->keys[thid] == startKey) {
                    rs = currNode->indices[thid];
                    break; // keys expected sorted; first match is enough
                }
            }
            recstart[bid] = rs;
        }

        // process leaves: find index of the ending record and length
        {
            knode *lastNode = &knodes[localLastKnode];
            int    rl       = reclength[bid]; // default fallback
            for (thid = 0; thid < threadsPerBlock; thid++) {
                if (lastNode->keys[thid] == endKey) {
                    rl = lastNode->indices[thid] - recstart[bid] + 1;
                    break;
                }
            }
            reclength[bid] = rl;
        }

        // write back updated node positions
        currKnode[bid] = localCurrKnode;
        lastKnode[bid] = localLastKnode;
        offset[bid]    = localOffset;
        offset_2[bid]  = localOffset_2;
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
