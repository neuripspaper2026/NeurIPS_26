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

    int threadsPerBlock;
    threadsPerBlock = order < 1024 ? order : 1024;

    time1 = get_time();

    // process number of queries
    for (int bid = 0; bid < count; bid++) {

        const int start_bid = start[bid];
        const int end_bid = end[bid];

        // process levels of the tree
        for (long level = 0; level < maxheight; level++) {

            long curr = currKnode[bid];
            long last = lastKnode[bid];

            const knode *curr_node = &knodes[curr];
            const knode *last_node = &knodes[last];

            long off = offset[bid];
            long off2 = offset_2[bid];

            // process all leaves at each level
            for (int thid = 0; thid < threadsPerBlock; thid++) {

                const int curr_k_this = curr_node->keys[thid];
                const int curr_k_next = curr_node->keys[thid + 1];

                if (curr_k_this <= start_bid && curr_k_next > start_bid) {

                    const long idx = curr_node->indices[thid];
                    if (idx < knodes_elem) {
                        off = idx;
                    }
                }

                const int last_k_this = last_node->keys[thid];
                const int last_k_next = last_node->keys[thid + 1];

                if (last_k_this <= end_bid && last_k_next > end_bid) {

                    const long idx2 = last_node->indices[thid];
                    if (idx2 < knodes_elem) {
                        off2 = idx2;
                    }
                }
            }

            // set for next tree level
            currKnode[bid] = off;
            lastKnode[bid] = off2;
            offset[bid] = off;
            offset_2[bid] = off2;
        }

        const long final_curr = currKnode[bid];
        const long final_last = lastKnode[bid];
        const knode *final_curr_node = &knodes[final_curr];
        const knode *final_last_node = &knodes[final_last];

        // process leaves - find start record index
        for (int thid = 0; thid < threadsPerBlock; thid++) {

            if (final_curr_node->keys[thid] == start_bid) {
                recstart[bid] = final_curr_node->indices[thid];
            }
        }

        // process leaves - find end record index and length
        const int rec_start_bid = recstart[bid];
        for (int thid = 0; thid < threadsPerBlock; thid++) {

            if (final_last_node->keys[thid] == end_bid) {
                reclength[bid] =
                    final_last_node->indices[thid] - rec_start_bid + 1;
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
