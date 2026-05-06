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

    // common variables
    int i;

    time0 = get_time();

    const int threadsPerBlock = order < 1024 ? order : 1024;

    time1 = get_time();

    // private thread IDs
    int thid;
    int bid;

    // process number of queries
    for (bid = 0; bid < count; ++bid) {

        const int start_bid = start[bid];
        const int end_bid   = end[bid];

        // process levels of the tree
        for (i = 0; i < maxheight; ++i) {

            const long curr_id = currKnode[bid];
            const long last_id = lastKnode[bid];

            const knode *restrict kn_curr = &knodes[curr_id];
            const knode *restrict kn_last = &knodes[last_id];

            long off1 = offset[bid];
            long off2 = offset_2[bid];

            // process all leaves at each level
            for (thid = 0; thid < threadsPerBlock; ++thid) {

                const int k0_s = kn_curr->keys[thid];
                const int k1_s = kn_curr->keys[thid + 1];

                if (k0_s <= start_bid && k1_s > start_bid) {

                    const long idx1 = kn_curr->indices[thid];
                    if (idx1 < knodes_elem) {
                        off1 = idx1;
                    }
                }

                const int k0_e = kn_last->keys[thid];
                const int k1_e = kn_last->keys[thid + 1];

                if (k0_e <= end_bid && k1_e > end_bid) {

                    const long idx2 = kn_last->indices[thid];
                    if (idx2 < knodes_elem) {
                        off2 = idx2;
                    }
                }
            }

            // set for next tree level
            currKnode[bid] = off1;
            lastKnode[bid] = off2;
            offset[bid] = off1;
            offset_2[bid] = off2;
        }

        const long final_curr = currKnode[bid];
        const long final_last = lastKnode[bid];

        const knode *restrict kn_final_curr = &knodes[final_curr];
        const knode *restrict kn_final_last = &knodes[final_last];

        // process leaves - starting record
        for (thid = 0; thid < threadsPerBlock; ++thid) {

            if (kn_final_curr->keys[thid] == start_bid) {
                recstart[bid] = kn_final_curr->indices[thid];
            }
        }

        const int rec_start_bid = recstart[bid];

        // process leaves - ending record / length
        for (thid = 0; thid < threadsPerBlock; ++thid) {

            if (kn_final_last->keys[thid] == end_bid) {
                reclength[bid] =
                    kn_final_last->indices[thid] - rec_start_bid + 1;
            }
        }
    }

    time2 = get_time();

    printf("Time spent in different stages of CPU/MCPU KERNEL:\n");

    const float total_time = (float)(time2 - time0);
    const float t_set = (float)(time1 - time0);
    const float t_kernel = (float)(time2 - time1);

    printf("%15.12f s, %15.12f % : MCPU: SET DEVICE\n",
           t_set / 1000000.0f,
           t_set / total_time * 100.0f);
    printf("%15.12f s, %15.12f % : CPU/MCPU: KERNEL\n",
           t_kernel / 1000000.0f,
           t_kernel / total_time * 100.0f);

    printf("Total time:\n");
    printf("%.12f s\n", total_time / 1000000.0f);

}
