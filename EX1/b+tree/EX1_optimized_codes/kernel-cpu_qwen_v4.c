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

    int threadsPerBlock;
    threadsPerBlock = order < 1024 ? order : 1024;

    time1 = get_time();

    // private thread IDs
    int thid;
    int bid;
    int i;

    // Cache frequently accessed pointers
    knode *current_knode_ptr;
    knode *offset_knode_ptr;
    
    // process number of queries
    for (bid = 0; bid < count; bid++) {
        long current_knode_idx = currKnode[bid];
        long offset_idx = offset[bid];
        int key = keys[bid];
        
        // process levels of the tree
        for (i = 0; i < maxheight; i++) {
            current_knode_ptr = &knodes[current_knode_idx];
            
            // process all leaves at each level
            for (thid = 0; thid < threadsPerBlock; thid++) {
                // if value is between the two keys
                if ((current_knode_ptr->keys[thid]) <= key &&
                    (current_knode_ptr->keys[thid + 1] > key)) {

                    if (current_knode_ptr->indices[thid] < knodes_elem) {
                        offset_idx = current_knode_ptr->indices[thid];
                        break; // Early exit since we found the matching node
                    }
                }
            }

            // set for next tree level
            current_knode_idx = offset_idx;
        }
        
        // Cache the final knode pointer for value retrieval
        current_knode_ptr = &knodes[current_knode_idx];
        
        for (thid = 0; thid < threadsPerBlock; thid++) {
            if (current_knode_ptr->keys[thid] == key) {
                ans[bid].value = records[current_knode_ptr->indices[thid]].value;
                break; // Early exit since we found the key
            }
        }
        
        // Update the arrays with final values
        currKnode[bid] = current_knode_idx;
        offset[bid] = offset_idx;
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
