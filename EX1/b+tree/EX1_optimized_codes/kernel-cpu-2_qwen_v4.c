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

    int threadsPerBlock;
    threadsPerBlock = order < 1024 ? order : 1024;

    time1 = get_time();

    // private thread IDs
    int thid;
    int bid;

    // Cache frequently accessed pointers
    knode *current_knode_ptr;
    knode *last_knode_ptr;
    
// process number of queries
    for (bid = 0; bid < count; bid++) {
        long current_knode_idx = currKnode[bid];
        long last_knode_idx = lastKnode[bid];
        long offset_idx = offset[bid];
        long offset_2_idx = offset_2[bid];
        int start_key = start[bid];
        int end_key = end[bid];

        // process levels of the tree
        for (i = 0; i < maxheight; i++) {
            current_knode_ptr = &knodes[current_knode_idx];
            last_knode_ptr = &knodes[last_knode_idx];

            // process all leaves at each level
            for (thid = 0; thid < threadsPerBlock; thid++) {
                if ((current_knode_ptr->keys[thid] <= start_key) &&
                    (current_knode_ptr->keys[thid + 1] > start_key)) {

                    if (current_knode_ptr->indices[thid] < knodes_elem) {
                        offset_idx = current_knode_ptr->indices[thid];
                    }
                }
                if ((last_knode_ptr->keys[thid] <= end_key) &&
                    (last_knode_ptr->keys[thid + 1] > end_key)) {

                    if (last_knode_ptr->indices[thid] < knodes_elem) {
                        offset_2_idx = last_knode_ptr->indices[thid];
                    }
                }
            }

            // set for next tree level
            current_knode_idx = offset_idx;
            last_knode_idx = offset_2_idx;
        }

        // Cache pointers for final processing
        current_knode_ptr = &knodes[current_knode_idx];
        last_knode_ptr = &knodes[last_knode_idx];
        
        // process leaves
        for (thid = 0; thid < threadsPerBlock; thid++) {
            // Find the index of the starting record
            if (current_knode_ptr->keys[thid] == start_key) {
                recstart[bid] = current_knode_ptr->indices[thid];
                break; // Early exit since we found the key
            }
        }

        // process leaves
        for (thid = 0; thid < threadsPerBlock; thid++) {
            // Find the index of the ending record
            if (last_knode_ptr->keys[thid] == end_key) {
                reclength[bid] = last_knode_ptr->indices[thid] - recstart[bid] + 1;
                break; // Early exit since we found the key
            }
        }
        
        // Update arrays with final values
        currKnode[bid] = current_knode_idx;
        lastKnode[bid] = last_knode_idx;
        offset[bid] = offset_idx;
        offset_2[bid] = offset_2_idx;
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
