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

    int threadsPerBlock = (order < 1024) ? order : 1024;

    time1 = get_time();

// process number of queries
    for (int bid = 0; bid < count; ++bid) {

        const int start_bid = start[bid];
        const int end_bid   = end[bid];

        // process levels of the tree
        for (long level = 0; level < maxheight; ++level) {

            const long curr_idx = currKnode[bid];
            const long last_idx = lastKnode[bid];

            knode *const curr_node = &knodes[curr_idx];
            knode *const last_node = &knodes[last_idx];

            long next_offset_start = offset[bid];
            long next_offset_end   = offset_2[bid];

            knode *const off_node_start = &knodes[next_offset_start];
            knode *const off_node_end   = &knodes[next_offset_end];

            // process all leaves at each level
            for (int thid = 0; thid < threadsPerBlock; ++thid) {

                const int key_curr      = curr_node->keys[thid];
                const int key_curr_next = curr_node->keys[thid + 1];

                if (key_curr <= start_bid && key_curr_next > start_bid) {

                    const long idx = off_node_start->indices[thid];

                    if (idx < knodes_elem) {
                        next_offset_start = idx;
                    }
                }

                const int key_last      = last_node->keys[thid];
                const int key_last_next = last_node->keys[thid + 1];

                if (key_last <= end_bid && key_last_next > end_bid) {

                    const long idx2 = off_node_end->indices[thid];

                    if (idx2 < knodes_elem) {
                        next_offset_end = idx2;
                    }
                }
            }

            // set for next tree level
            offset[bid]   = next_offset_start;
            offset_2[bid] = next_offset_end;

            currKnode[bid] = next_offset_start;
            lastKnode[bid] = next_offset_end;
        }

        const long final_curr_idx = currKnode[bid];
        const long final_last_idx = lastKnode[bid];

        knode *const final_curr_node = &knodes[final_curr_idx];
        knode *const final_last_node = &knodes[final_last_idx];

        // process leaves - Find the index of the starting record
        for (int thid = 0; thid < threadsPerBlock; ++thid) {

            if (final_curr_node->keys[thid] == start_bid) {
                recstart[bid] = final_curr_node->indices[thid];
            }
        }

        const int recstart_bid = recstart[bid];

        // process leaves - Find the index of the ending record
        for (int thid = 0; thid < threadsPerBlock; ++thid) {

            if (final_last_node->keys[thid] == end_bid) {
                reclength[bid] =
                    final_last_node->indices[thid] - recstart_bid + 1;
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
