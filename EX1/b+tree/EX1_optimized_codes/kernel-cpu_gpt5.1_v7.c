#include <stdlib.h> // (in directory known to compiler)			needed by malloc
#include <stdio.h> // (in directory known to compiler)			needed by printf, stderr

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

    const int threadsPerBlock = order < 1024 ? order : 1024;

    time1 = get_time();

    // private thread IDs
    int thid;
    int bid;
    long level;

    // process number of queries
    for (bid = 0; bid < count; ++bid) {

        const int key_bid = keys[bid];

        // process levels of the tree
        for (level = 0; level < maxheight; ++level) {

            const long curr = currKnode[bid];
            const knode *restrict kn_curr = &knodes[curr];
            long off = offset[bid];

            // process all leaves at each level
            for (thid = 0; thid < threadsPerBlock; ++thid) {

                const int k0 = kn_curr->keys[thid];
                const int k1 = kn_curr->keys[thid + 1];

                // if value is between the two keys
                if (k0 <= key_bid && k1 > key_bid) {

                    const long idx = knodes[off].indices[thid];
                    if (idx < knodes_elem) {
                        off = idx;
                    }
                }
            }

            // set for next tree level
            offset[bid] = off;
            currKnode[bid] = off;
        }

        const long final_curr = currKnode[bid];
        const knode *restrict kn_final = &knodes[final_curr];

        for (thid = 0; thid < threadsPerBlock; ++thid) {

            if (kn_final->keys[thid] == key_bid) {
                ans[bid].value =
                    records[kn_final->indices[thid]].value;
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
