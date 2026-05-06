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
#pragma omp parallel for default(none) private(bid, thid, i) shared(count, maxheight, threadsPerBlock,                     \
                                                                   knodes, knodes_elem, currKnode, offset, lastKnode,     \
                                                                   offset_2, start, end, recstart, reclength)
#endif
    for (bid = 0; bid < count; bid++) {

        long curr = currKnode[bid];
        long last = lastKnode[bid];
        long off = offset[bid];
        long off2 = offset_2[bid];
        const int s = start[bid];
        const int e = end[bid];

        // process levels of the tree
        for (i = 0; i < maxheight; i++) {

            const knode *curr_node = &knodes[curr];
            const knode *last_node = &knodes[last];

            // process all leaves at each level
            for (thid = 0; thid < threadsPerBlock; thid++) {

                const int curr_key = curr_node->keys[thid];
                const int curr_next_key = curr_node->keys[thid + 1];

                if (curr_key <= s && curr_next_key > s) {

                    long idx = curr_node->indices[thid];
                    if (idx < knodes_elem) {
                        off = idx;
                    }
                }

                const int last_key = last_node->keys[thid];
                const int last_next_key = last_node->keys[thid + 1];

                if (last_key <= e && last_next_key > e) {

                    long idx2 = last_node->indices[thid];
                    if (idx2 < knodes_elem) {
                        off2 = idx2;
                    }
                }
            }

            // set for next tree level
            curr = off;
            last = off2;
        }

        currKnode[bid] = curr;
        lastKnode[bid] = last;
        offset[bid] = off;
        offset_2[bid] = off2;

        const knode *leaf_start = &knodes[curr];
        const knode *leaf_end = &knodes[last];

        // process leaves - find the index of the starting record
        for (thid = 0; thid < threadsPerBlock; thid++) {

            if (leaf_start->keys[thid] == s) {
                recstart[bid] = leaf_start->indices[thid];
            }
        }

        // process leaves - find the index of the ending record
        for (thid = 0; thid < threadsPerBlock; thid++) {

            if (leaf_end->keys[thid] == e) {
                reclength[bid] =
                    leaf_end->indices[thid] - recstart[bid] + 1;
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
