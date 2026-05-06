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
#pragma omp parallel for private(thid, i) schedule(static)
#endif
    for (bid = 0; bid < count; bid++) {

        long curr  = currKnode[bid];
        long last  = lastKnode[bid];
        long off   = offset[bid];
        long off2  = offset_2[bid];
        const int s = start[bid];
        const int e = end[bid];

        // process levels of the tree
        for (i = 0; i < maxheight; i++) {

            knode *curr_node = &knodes[curr];
            knode *last_node = &knodes[last];

            // process all leaves at each level
            for (thid = 0; thid < threadsPerBlock; thid++) {

                int key_curr_left  = curr_node->keys[thid];
                int key_curr_right = curr_node->keys[thid + 1];

                if (key_curr_left <= s && key_curr_right > s) {

                    long idx = curr_node->indices[thid];
                    if (idx < knodes_elem) {
                        off = idx;
                    }
                }

                int key_last_left  = last_node->keys[thid];
                int key_last_right = last_node->keys[thid + 1];

                if (key_last_left <= e && key_last_right > e) {

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

        currKnode[bid]  = curr;
        lastKnode[bid]  = last;
        offset[bid]     = off;
        offset_2[bid]   = off2;

        knode *curr_leaf = &knodes[curr];
        knode *last_leaf = &knodes[last];

        // process leaves: find the index of the starting record
        for (thid = 0; thid < threadsPerBlock; thid++) {

            if (curr_leaf->keys[thid] == s) {
                recstart[bid] = curr_leaf->indices[thid];
                break;
            }
        }

        // process leaves: find the index of the ending record
        for (thid = 0; thid < threadsPerBlock; thid++) {

            if (last_leaf->keys[thid] == e) {
                reclength[bid] =
                    last_leaf->indices[thid] - recstart[bid] + 1;
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
