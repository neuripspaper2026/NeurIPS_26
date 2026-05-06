#include <stdlib.h> // (in directory known to compiler)
#include <stdio.h> // (in directory known to compiler)                  needed by printf

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

    time0 = get_time();

    const int threadsPerBlock = order < 1024 ? order : 1024;

    time1 = get_time();

// process number of queries
    for (int bid = 0; bid < count; ++bid) {

        const int s = start[bid];
        const int e = end[bid];

        long curr = currKnode[bid];
        long last = lastKnode[bid];
        long off = offset[bid];
        long off2 = offset_2[bid];

        // process levels of the tree
        for (long level = 0; level < maxheight; ++level) {

            knode *const curr_node = &knodes[curr];
            knode *const last_node = &knodes[last];

            const int *const curr_keys = curr_node->keys;
            const int *const last_keys = last_node->keys;
            const long *const curr_indices = curr_node->indices;
            const long *const last_indices = last_node->indices;

            // process all leaves at each level
            for (int thid = 0; thid < threadsPerBlock; ++thid) {

                const int ckey = curr_keys[thid];
                const int ckey_next = curr_keys[thid + 1];

                if (ckey <= s && ckey_next > s) {
                    const long idx = curr_indices[thid];
                    if (idx < knodes_elem) {
                        off = idx;
                    }
                }

                const int lkey = last_keys[thid];
                const int lkey_next = last_keys[thid + 1];

                if (lkey <= e && lkey_next > e) {
                    const long idx2 = last_indices[thid];
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

        knode *const curr_leaf = &knodes[curr];
        knode *const last_leaf = &knodes[last];

        const int *const curr_leaf_keys = curr_leaf->keys;
        const long *const curr_leaf_indices = curr_leaf->indices;
        const int *const last_leaf_keys = last_leaf->keys;
        const long *const last_leaf_indices = last_leaf->indices;

        // process leaves - find starting record
        for (int thid = 0; thid < threadsPerBlock; ++thid) {

            if (curr_leaf_keys[thid] == s) {
                recstart[bid] = curr_leaf_indices[thid];
            }
        }

        // process leaves - find ending record
        for (int thid = 0; thid < threadsPerBlock; ++thid) {

            if (last_leaf_keys[thid] == e) {
                reclength[bid] =
                    last_leaf_indices[thid] - recstart[bid] + 1;
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
