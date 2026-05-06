#include <stdlib.h> // (in directory known to compiler)
#include <stdio.h>  // (in directory known to compiler) needed by printf

#include "../common.h" // (in directory provided here)

#include "../util/timer/timer.h" // (in directory provided here) needed by timer

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

    // private thread IDs
    int thid;
    int bid;
    long level;

    // process number of queries
    for (bid = 0; bid < count; ++bid) {

        const int s_key = start[bid];
        const int e_key = end[bid];

        // process levels of the tree
        for (level = 0; level < maxheight; ++level) {

            const long curr_idx  = currKnode[bid];
            const long last_idx  = lastKnode[bid];
            const long off_idx   = offset[bid];
            const long off2_idx  = offset_2[bid];

            knode *const curr_node = &knodes[curr_idx];
            knode *const last_node = &knodes[last_idx];
            knode *const off_node  = &knodes[off_idx];
            knode *const off2_node = &knodes[off2_idx];

            int found_start = 0;
            int found_end   = 0;

            // process all leaves at each level
            for (thid = 0; thid < threadsPerBlock; ++thid) {

                // start boundary search
                if (!found_start) {
                    const int key_left_s  = curr_node->keys[thid];
                    const int key_right_s = curr_node->keys[thid + 1];

                    if (key_left_s <= s_key && key_right_s > s_key) {
                        const long idx_s = off_node->indices[thid];
                        if (idx_s < knodes_elem) {
                            offset[bid] = idx_s;
                        }
                        found_start = 1;
                    }
                }

                // end boundary search
                if (!found_end) {
                    const int key_left_e  = last_node->keys[thid];
                    const int key_right_e = last_node->keys[thid + 1];

                    if (key_left_e <= e_key && key_right_e > e_key) {
                        const long idx_e = off2_node->indices[thid];
                        if (idx_e < knodes_elem) {
                            offset_2[bid] = idx_e;
                        }
                        found_end = 1;
                    }
                }

                if (found_start && found_end) {
                    break;
                }
            }

            // set for next tree level
            currKnode[bid] = offset[bid];
            lastKnode[bid] = offset_2[bid];
        }

        // process leaves - find starting record index
        {
            const long leaf_start_idx = currKnode[bid];
            knode *const leaf_start   = &knodes[leaf_start_idx];

            for (thid = 0; thid < threadsPerBlock; ++thid) {

                if (leaf_start->keys[thid] == s_key) {
                    recstart[bid] = leaf_start->indices[thid];
                    break;
                }
            }
        }

        // process leaves - find ending record index and compute length
        {
            const long leaf_end_idx = lastKnode[bid];
            knode *const leaf_end   = &knodes[leaf_end_idx];

            for (thid = 0; thid < threadsPerBlock; ++thid) {

                if (leaf_end->keys[thid] == e_key) {
                    reclength[bid] =
                        leaf_end->indices[thid] - recstart[bid] + 1;
                    break;
                }
            }
        }
    }

    time2 = get_time();

    printf("Time spent in different stages of CPU/MCPU KERNEL:\n");

    printf("%15.12f s, %15.12f % : MCPU: SET DEVICE\n",
           (float)(time1 - time0) / 1000000.0f,
           (float)(time1 - time0) / (float)(time2 - time0) * 100.0f);
    printf("%15.12f s, %15.12f % : CPU/MCPU: KERNEL\n",
           (float)(time2 - time1) / 1000000.0f,
           (float)(time2 - time1) / (float)(time2 - time0) * 100.0f);

    printf("Total time:\n");
    printf("%.12f s\n", (float)(time2 - time0) / 1000000.0f);
}
