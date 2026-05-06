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

    time0 = get_time();

    int threadsPerBlock;
    threadsPerBlock = order < 1024 ? order : 1024;

    time1 = get_time();

    // process number of queries
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(count, maxheight, threadsPerBlock, knodes, knodes_elem, currKnode, lastKnode, offset, offset_2, start, end, recstart, reclength) schedule(static)
#endif
    for (int bid = 0; bid < count; bid++) {

        long curr = currKnode[bid];
        long last = lastKnode[bid];
        long off_curr = offset[bid];
        long off_last = offset_2[bid];
        int s = start[bid];
        int e = end[bid];

        // process levels of the tree
        for (long level = 0; level < maxheight; level++) {

            knode *curr_node = &knodes[curr];
            knode *last_node = &knodes[last];

            // process all leaves at each level
            for (int thid = 0; thid < threadsPerBlock; thid++) {

                int k_curr_l = curr_node->keys[thid];
                int k_curr_r = curr_node->keys[thid + 1];

                if (k_curr_l <= s && k_curr_r > s) {

                    long idx = curr_node->indices[thid];
                    if (idx < knodes_elem) {
                        off_curr = idx;
                    }
                }

                int k_last_l = last_node->keys[thid];
                int k_last_r = last_node->keys[thid + 1];

                if (k_last_l <= e && k_last_r > e) {

                    long idx2 = last_node->indices[thid];
                    if (idx2 < knodes_elem) {
                        off_last = idx2;
                    }
                }
            }

            // set for next tree level
            curr = off_curr;
            last = off_last;
        }

        currKnode[bid] = curr;
        lastKnode[bid] = last;
        offset[bid] = off_curr;
        offset_2[bid] = off_last;

        knode *curr_node_final = &knodes[curr];
        knode *last_node_final = &knodes[last];

        // Find the index of the starting record
        int rec_s = recstart[bid];
        for (int thid = 0; thid < threadsPerBlock; thid++) {

            if (curr_node_final->keys[thid] == s) {
                rec_s = curr_node_final->indices[thid];
            }
        }
        recstart[bid] = rec_s;

        // Find the index of the ending record
        int rec_len = reclength[bid];
        for (int thid = 0; thid < threadsPerBlock; thid++) {

            if (last_node_final->keys[thid] == e) {
                rec_len = last_node_final->indices[thid] - rec_s + 1;
            }
        }
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
