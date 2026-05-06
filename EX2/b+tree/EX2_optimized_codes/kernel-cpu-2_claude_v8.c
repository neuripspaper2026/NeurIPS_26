#include <stdlib.h>
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "../common.h"
#include "../util/timer/timer.h"
#include "../kernel_cpu_2.h"

void kernel_cpu_2(knode *knodes, long knodes_elem,

                  int order, long maxheight, int count,

                  long *currKnode, long *offset, long *lastKnode,
                  long *offset_2, int *start, int *end, int *recstart,
                  int *reclength) {

    long long time0;
    long long time1;
    long long time2;

    int i;

    time0 = get_time();

    int threadsPerBlock;
    threadsPerBlock = order < 1024 ? order : 1024;

    time1 = get_time();

    #pragma omp parallel
    {
        int bid, thid;
        long local_currKnode, local_lastKnode, local_offset, local_offset_2;
        int local_recstart, local_reclength;
        
        #pragma omp for schedule(dynamic, 1)
        for (bid = 0; bid < count; bid++) {
            local_currKnode = currKnode[bid];
            local_lastKnode = lastKnode[bid];
            local_offset = offset[bid];
            local_offset_2 = offset_2[bid];
            
            int start_key = start[bid];
            int end_key = end[bid];
            
            for (i = 0; i < maxheight; i++) {
                knode *curr_node = &knodes[local_currKnode];
                knode *last_node = &knodes[local_lastKnode];
                
                for (thid = 0; thid < threadsPerBlock; thid++) {
                    if ((curr_node->keys[thid] <= start_key) &&
                        (curr_node->keys[thid + 1] > start_key)) {
                        long idx = curr_node->indices[thid];
                        if (idx < knodes_elem) {
                            local_offset = idx;
                        }
                    }
                    if ((last_node->keys[thid] <= end_key) &&
                        (last_node->keys[thid + 1] > end_key)) {
                        long idx = last_node->indices[thid];
                        if (idx < knodes_elem) {
                            local_offset_2 = idx;
                        }
                    }
                }
                
                local_currKnode = local_offset;
                local_lastKnode = local_offset_2;
            }
            
            knode *curr_leaf = &knodes[local_currKnode];
            knode *last_leaf = &knodes[local_lastKnode];
            
            local_recstart = 0;
            local_reclength = 0;
            
            for (thid = 0; thid < threadsPerBlock; thid++) {
                if (curr_leaf->keys[thid] == start_key) {
                    local_recstart = curr_leaf->indices[thid];
                }
            }
            
            for (thid = 0; thid < threadsPerBlock; thid++) {
                if (last_leaf->keys[thid] == end_key) {
                    local_reclength = last_leaf->indices[thid] - local_recstart + 1;
                }
            }
            
            currKnode[bid] = local_currKnode;
            lastKnode[bid] = local_lastKnode;
            offset[bid] = local_offset;
            offset_2[bid] = local_offset_2;
            recstart[bid] = local_recstart;
            reclength[bid] = local_reclength;
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
