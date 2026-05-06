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
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int bid = 0; bid < count; bid++) {
        long curr_knode = currKnode[bid];
        long off = offset[bid];
        int key = keys[bid];

        // process levels of the tree
        for (int i = 0; i < maxheight; i++) {
            int thid;
            // process all leaves at each level
            for (thid = 0; thid < threadsPerBlock - 1; thid++) {
                // if value is between the two keys
                if ((knodes[curr_knode].keys[thid]) <= key &&
                    (knodes[curr_knode].keys[thid + 1] > key)) {
                    if (knodes[off].indices[thid] < knodes_elem) {
                        off = knodes[off].indices[thid];
                    }
                    break; // Found the matching child, exit inner loop early
                }
            }

            // set for next tree level
            curr_knode = off;
        }

        // Final search for the key
        for (int thid = 0; thid < threadsPerBlock; thid++) {
            if (knodes[curr_knode].keys[thid] == key) {
                ans[bid].value = records[knodes[curr_knode].indices[thid]].value;
                break; // Found the key, exit loop early
            }
        }
        
        // Update the global arrays with final values
        currKnode[bid] = curr_knode;
        offset[bid] = off;
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
