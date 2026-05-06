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

    // process number of queries
    #pragma omp parallel for schedule(dynamic)
    for (int bid = 0; bid < count; bid++) {
        // private thread IDs
        int thid;
        int i;

        // process levels of the tree
        for (i = 0; i < maxheight; i++) {
            long curr_knode_val = currKnode[bid];
            long offset_val = offset[bid];

            // process all leaves at each level
            for (thid = 0; thid < threadsPerBlock - 1; thid++) {
                // if value is between the two keys
                if ((knodes[curr_knode_val].keys[thid]) <= keys[bid] &&
                    (knodes[curr_knode_val].keys[thid + 1] > keys[bid])) {
                    if (knodes[offset_val].indices[thid] < knodes_elem) {
                        offset_val = knodes[offset_val].indices[thid];
                    }
                }
            }

            // set for next tree level
            currKnode[bid] = offset_val;
            offset[bid] = offset_val;
        }

        long final_knode = currKnode[bid];
        // Final search for matching key
        for (thid = 0; thid < threadsPerBlock; thid++) {
            if (knodes[final_knode].keys[thid] == keys[bid]) {
                ans[bid].value = records[knodes[final_knode].indices[thid]].value;
                break; // Early exit once found
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
