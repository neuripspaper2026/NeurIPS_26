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
#pragma omp parallel for private(thid, i) schedule(static)
#endif
    for (bid = 0; bid < count; bid++) {

        long curr = currKnode[bid];
        long off = offset[bid];
        const int key = keys[bid];

        // process levels of the tree
        for (i = 0; i < maxheight; i++) {

            // process all leaves at each level
            knode *curr_node = &knodes[curr];
            knode *off_node  = &knodes[off];

            for (thid = 0; thid < threadsPerBlock; thid++) {

                // if value is between the two keys
                int key_left  = curr_node->keys[thid];
                int key_right = curr_node->keys[thid + 1];

                if (key_left <= key && key_right > key) {

                    long idx = off_node->indices[thid];
                    if (idx < knodes_elem) {
                        off = idx;
                        off_node = &knodes[off];
                    }
                }
            }

            // set for next tree level
            curr = off;
            curr_node = &knodes[curr];
        }

        currKnode[bid] = curr;
        offset[bid]    = off;

        knode *final_node = &knodes[curr];
        // search within the final node
        for (thid = 0; thid < threadsPerBlock; thid++) {

            if (final_node->keys[thid] == key) {
                ans[bid].value = records[final_node->indices[thid]].value;
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
