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
        long local_offset;
        long local_currKnode;
        int local_key;

        #pragma omp for schedule(dynamic, 1)
        for (bid = 0; bid < count; bid++) {
            local_offset = offset[bid];
            local_currKnode = currKnode[bid];
            local_key = keys[bid];

            for (i = 0; i < maxheight; i++) {
                knode *curr_node = &knodes[local_currKnode];
                int *node_keys = curr_node->keys;
                long *node_indices = curr_node->indices;
                
                knode *offset_node = &knodes[local_offset];
                long *offset_indices = offset_node->indices;

                for (thid = 0; thid < threadsPerBlock; thid++) {
                    if (node_keys[thid] <= local_key && node_keys[thid + 1] > local_key) {
                        long idx = offset_indices[thid];
                        if (idx < knodes_elem) {
                            local_offset = idx;
                        }
                    }
                }

                local_currKnode = local_offset;
            }

            knode *final_node = &knodes[local_currKnode];
            int *final_keys = final_node->keys;
            long *final_indices = final_node->indices;

            for (thid = 0; thid < threadsPerBlock; thid++) {
                if (final_keys[thid] == local_key) {
                    ans[bid].value = records[final_indices[thid]].value;
                    break;
                }
            }

            currKnode[bid] = local_currKnode;
            offset[bid] = local_offset;
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
