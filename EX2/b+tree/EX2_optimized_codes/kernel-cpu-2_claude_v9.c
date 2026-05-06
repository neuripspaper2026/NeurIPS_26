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
        long local_currKnode, local_lastKnode;
        long local_offset, local_offset_2;
        int local_start, local_end;
        int local_recstart, local_reclength;

        #pragma omp for schedule(dynamic, 1)
        for (bid = 0; bid < count; bid++) {
            local_currKnode = currKnode[bid];
            local_lastKnode = lastKnode[bid];
            local_offset = offset[bid];
            local_offset_2 = offset_2[bid];
            local_start = start[bid];
            local_end = end[bid];

            for (i = 0; i < maxheight; i++) {
                knode *curr_node = &knodes[local_currKnode];
                knode *last_node = &knodes[local_lastKnode];
                int *curr_keys = curr_node->keys;
                int *last_keys = last_node->keys;
                long *curr_indices = curr_node->indices;
                long *last_indices = last_node->indices;

                for (thid = 0; thid < threadsPerBlock; thid++) {
                    if (curr_keys[thid] <= local_start && curr_keys[thid + 1] > local_start) {
                        long idx = curr_indices[thid];
                        if (idx < knodes_elem) {
                            local_offset = idx;
                        }
                        break;
                    }
                }

                for (thid = 0; thid < threadsPerBlock; thid++) {
                    if (last_keys[thid] <= local_end && last_keys[thid + 1] > local_end) {
                        long idx = last_indices[thid];
                        if (idx < knodes_elem) {
                            local_offset_2 = idx;
                        }
                        break;
                    }
                }

                local_currKnode = local_offset;
                local_lastKnode = local_offset_2;
            }

            knode *final_curr_node = &knodes[local_currKnode];
            knode *final_last_node = &knodes[local_lastKnode];
            int *final_curr_keys = final_curr_node->keys;
            int *final_last_keys = final_last_node->keys;
            long *final_curr_indices = final_curr_node->indices;
            long *final_last_indices = final_last_node->indices;

            for (thid = 0; thid < threadsPerBlock; thid++) {
                if (final_curr_keys[thid] == local_start) {
                    local_recstart = final_curr_indices[thid];
                    break;
                }
            }

            for (thid = 0; thid < threadsPerBlock; thid++) {
                if (final_last_keys[thid] == local_end) {
                    local_reclength = final_last_indices[thid] - local_recstart + 1;
                    break;
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
