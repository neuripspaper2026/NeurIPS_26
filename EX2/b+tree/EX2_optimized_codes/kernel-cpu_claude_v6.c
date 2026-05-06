#include <stdlib.h>
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "../common.h"
#include "../util/timer/timer.h"
#include "../kernel_cpu.h"

void kernel_cpu(record *records, knode *knodes, long knodes_elem,
                int order, long maxheight, int count,
                long *currKnode, long *offset, int *keys, record *ans) {

    long long time0;
    long long time1;
    long long time2;

    time0 = get_time();

    int threadsPerBlock;
    threadsPerBlock = order < 1024 ? order : 1024;

    time1 = get_time();

    #pragma omp parallel
    {
        int bid, i, thid;
        long local_offset, local_currKnode;
        
        #pragma omp for schedule(dynamic, 1) nowait
        for (bid = 0; bid < count; bid++) {
            local_currKnode = currKnode[bid];
            local_offset = offset[bid];
            
            for (i = 0; i < maxheight; i++) {
                knode *curr_node = &knodes[local_currKnode];
                knode *offset_node = &knodes[local_offset];
                int key = keys[bid];
                
                for (thid = 0; thid < threadsPerBlock; thid++) {
                    if (curr_node->keys[thid] <= key && curr_node->keys[thid + 1] > key) {
                        long idx = offset_node->indices[thid];
                        if (idx < knodes_elem) {
                            local_offset = idx;
                            offset_node = &knodes[local_offset];
                        }
                    }
                }
                
                local_currKnode = local_offset;
            }
            
            knode *final_node = &knodes[local_currKnode];
            int key = keys[bid];
            
            for (thid = 0; thid < threadsPerBlock; thid++) {
                if (final_node->keys[thid] == key) {
                    ans[bid].value = records[final_node->indices[thid]].value;
                    break;
                }
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
