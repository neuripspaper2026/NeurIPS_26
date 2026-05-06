#include <stdlib.h> // (in directory known to compiler)			needed by malloc
#include <stdio.h>  // (in directory known to compiler)			needed by printf, stderr
#ifdef _OPENMP
#include <omp.h>
#endif

#include "../common.h" // (in directory provided here)

#include "../util/timer/timer.h" // (in directory provided here)
#include "../kernel_cpu.h"        // (in directory provided here)

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

    int bid;
    int thid;
    long i;

// process number of queries
#ifdef _OPENMP
#pragma omp parallel for default(none) private(bid, thid, i) shared(count, maxheight, threadsPerBlock,                     \
                                                                   currKnode, offset, knodes, knodes_elem, keys,         \
                                                                   records, ans)
#endif
    for (bid = 0; bid < count; bid++) {

        long curr = currKnode[bid];
        long off = offset[bid];

// process levels of the tree
        for (i = 0; i < maxheight; i++) {

            // process all leaves at each level
            for (thid = 0; thid < threadsPerBlock; thid++) {

                const int key = keys[bid];
                const knode *curr_node = &knodes[curr];
                const int curr_key = curr_node->keys[thid];
                const int next_key = curr_node->keys[thid + 1];

                // if value is between the two keys
                if (curr_key <= key && next_key > key) {
                    const knode *off_node = &knodes[off];
                    long idx = off_node->indices[thid];
                    if (idx < knodes_elem) {
                        off = idx;
                    }
                }
            }

            // set for next tree level
            curr = off;
        }

        currKnode[bid] = curr;
        offset[bid] = off;

        for (thid = 0; thid < threadsPerBlock; thid++) {
            const knode *curr_node = &knodes[curr];
            if (curr_node->keys[thid] == keys[bid]) {
                ans[bid].value = records[curr_node->indices[thid]].value;
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
