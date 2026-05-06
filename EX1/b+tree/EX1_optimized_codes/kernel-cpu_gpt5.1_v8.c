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

    int threadsPerBlock = order < 1024 ? order : 1024;

    time1 = get_time();

    // private thread IDs
    int thid;
    int bid;
    long i;

    knode *knodes_local = knodes;
    record *records_local = records;

    // process number of queries
    for (bid = 0; bid < count; ++bid) {

        const int key_bid = keys[bid];
        long curr_bid = currKnode[bid];
        long off_bid = offset[bid];

        // process levels of the tree
        for (i = 0; i < maxheight; ++i) {

            const knode *curr_node = &knodes_local[curr_bid];
            const knode *off_node = &knodes_local[off_bid];
            int last_key = curr_node->keys[0];

            // process all leaves at each level
            for (thid = 0; thid < threadsPerBlock; ++thid) {

                const int curr_key = curr_node->keys[thid];
                const int next_key = curr_node->keys[thid + 1];

                // if value is between the two keys
                if (curr_key <= key_bid && next_key > key_bid) {
                    const long idx = off_node->indices[thid];
                    if (idx < knodes_elem) {
                        off_bid = idx;
                        off_node = &knodes_local[off_bid];
                    }
                }

                last_key = curr_key;
            }

            // set for next tree level
            curr_bid = off_bid;
        }

        currKnode[bid] = curr_bid;
        offset[bid] = off_bid;

        const knode *final_node = &knodes_local[curr_bid];

        for (thid = 0; thid < threadsPerBlock; ++thid) {

            if (final_node->keys[thid] == key_bid) {
                ans[bid].value =
                    records_local[final_node->indices[thid]].value;
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
