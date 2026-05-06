#include <stdlib.h> // (in directory known to compiler)
#include <stdio.h>  // (in directory known to compiler) needed by printf

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

    int threadsPerBlock = order < 1024 ? order : 1024;

    time1 = get_time();

    // private thread IDs
    int thid;
    int bid;
    long i;

    knode *knodes_local = knodes;

    // process number of queries
    for (bid = 0; bid < count; ++bid) {

        const int start_bid = start[bid];
        const int end_bid = end[bid];

        long curr_bid = currKnode[bid];
        long last_bid = lastKnode[bid];
        long off_bid = offset[bid];
        long off2_bid = offset_2[bid];

        // process levels of the tree
        for (i = 0; i < maxheight; ++i) {

            const knode *curr_node = &knodes_local[curr_bid];
            const knode *last_node = &knodes_local[last_bid];

            // process all leaves at each level
            for (thid = 0; thid < threadsPerBlock; ++thid) {

                const int curr_key = curr_node->keys[thid];
                const int curr_next_key = curr_node->keys[thid + 1];

                if (curr_key <= start_bid && curr_next_key > start_bid) {
                    const long idx = curr_node->indices[thid];
                    if (idx < knodes_elem) {
                        off_bid = idx;
                    }
                }

                const int last_key = last_node->keys[thid];
                const int last_next_key = last_node->keys[thid + 1];

                if (last_key <= end_bid && last_next_key > end_bid) {
                    const long idx2 = last_node->indices[thid];
                    if (idx2 < knodes_elem) {
                        off2_bid = idx2;
                    }
                }
            }

            // set for next tree level
            curr_bid = off_bid;
            last_bid = off2_bid;
        }

        currKnode[bid] = curr_bid;
        lastKnode[bid] = last_bid;
        offset[bid] = off_bid;
        offset_2[bid] = off2_bid;

        const knode *curr_leaf = &knodes_local[curr_bid];
        const knode *last_leaf = &knodes_local[last_bid];

        // Find the index of the starting record
        for (thid = 0; thid < threadsPerBlock; ++thid) {
            if (curr_leaf->keys[thid] == start_bid) {
                recstart[bid] = curr_leaf->indices[thid];
            }
        }

        // Find the index of the ending record
        for (thid = 0; thid < threadsPerBlock; ++thid) {
            if (last_leaf->keys[thid] == end_bid) {
                reclength[bid] =
                    last_leaf->indices[thid] - recstart[bid] + 1;
            }
        }
    }

    time2 = get_time();

    const float total_time = (float)(time2 - time0);
    const float set_device_time = (float)(time1 - time0);
    const float kernel_time = (float)(time2 - time1);
    const float inv_total_time = 1.0f / total_time;

    printf("Time spent in different stages of CPU/MCPU KERNEL:\n");

    printf("%15.12f s, %15.12f % : MCPU: SET DEVICE\n",
           set_device_time / 1000000.0f,
           set_device_time * inv_total_time * 100.0f);
    printf("%15.12f s, %15.12f % : CPU/MCPU: KERNEL\n",
           kernel_time / 1000000.0f,
           kernel_time * inv_total_time * 100.0f);

    printf("Total time:\n");
    printf("%.12f s\n", total_time / 1000000.0f);
}
