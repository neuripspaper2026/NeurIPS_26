#include <stdlib.h> // (in directory known to compiler)
#include <stdio.h> // (in directory known to compiler)                  needed by printf
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

    // process number of queries
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int bid = 0; bid < count; bid++) {
        // Private copies for each thread
        long curr_knode = currKnode[bid];
        long last_knode = lastKnode[bid];
        long off = offset[bid];
        long off2 = offset_2[bid];
        int start_key = start[bid];
        int end_key = end[bid];
        int rec_start = recstart[bid];
        int rec_len = reclength[bid];

        // process levels of the tree
        for (i = 0; i < maxheight; i++) {
            int thid;

            // process all leaves at each level
            for (thid = 0; thid < threadsPerBlock - 1; thid++) {
                // Check start key
                if ((knodes[curr_knode].keys[thid] <= start_key) &&
                    (knodes[curr_knode].keys[thid + 1] > start_key)) {
                    if (knodes[curr_knode].indices[thid] < knodes_elem) {
                        off = knodes[curr_knode].indices[thid];
                    }
                    break; // Found the matching child, exit inner loop early
                }
            }

            // process all leaves at each level for end key
            for (thid = 0; thid < threadsPerBlock - 1; thid++) {
                // Check end key
                if ((knodes[last_knode].keys[thid] <= end_key) &&
                    (knodes[last_knode].keys[thid + 1] > end_key)) {
                    if (knodes[last_knode].indices[thid] < knodes_elem) {
                        off2 = knodes[last_knode].indices[thid];
                    }
                    break; // Found the matching child, exit inner loop early
                }
            }

            // set for next tree level
            curr_knode = off;
            last_knode = off2;
        }

        // process leaves for start key
        for (int thid = 0; thid < threadsPerBlock; thid++) {
            // Find the index of the starting record
            if (knodes[curr_knode].keys[thid] == start_key) {
                rec_start = knodes[curr_knode].indices[thid];
                break; // Found the key, exit loop early
            }
        }

        // process leaves for end key
        for (int thid = 0; thid < threadsPerBlock; thid++) {
            // Find the index of the ending record
            if (knodes[last_knode].keys[thid] == end_key) {
                rec_len = knodes[last_knode].indices[thid] - rec_start + 1;
                break; // Found the key, exit loop early
            }
        }

        // Update the global arrays with final values
        currKnode[bid] = curr_knode;
        lastKnode[bid] = last_knode;
        offset[bid] = off;
        offset_2[bid] = off2;
        recstart[bid] = rec_start;
        reclength[bid] = rec_len;
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
