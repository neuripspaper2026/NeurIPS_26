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

    time0 = get_time();

    int threadsPerBlock;
    threadsPerBlock = order < 1024 ? order : 1024;

    time1 = get_time();

#ifdef _OPENMP
    #pragma omp parallel
    {
        int bid, i, thid;
        
        #pragma omp for schedule(dynamic, 1)
        for (bid = 0; bid < count; bid++) {
            long curr = currKnode[bid];
            long last = lastKnode[bid];
            long off = offset[bid];
            long off2 = offset_2[bid];
            int start_val = start[bid];
            int end_val = end[bid];
            
            for (i = 0; i < maxheight; i++) {
                long next_offset = off;
                long next_offset2 = off2;
                
                for (thid = 0; thid < threadsPerBlock; thid++) {
                    int lower_key = knodes[curr].keys[thid];
                    int upper_key = knodes[curr].keys[thid + 1];
                    
                    if (lower_key <= start_val && upper_key > start_val) {
                        long idx = knodes[curr].indices[thid];
                        if (idx < knodes_elem) {
                            next_offset = idx;
                        }
                    }
                }
                
                for (thid = 0; thid < threadsPerBlock; thid++) {
                    int lower_key = knodes[last].keys[thid];
                    int upper_key = knodes[last].keys[thid + 1];
                    
                    if (lower_key <= end_val && upper_key > end_val) {
                        long idx = knodes[last].indices[thid];
                        if (idx < knodes_elem) {
                            next_offset2 = idx;
                        }
                    }
                }
                
                off = next_offset;
                off2 = next_offset2;
                curr = off;
                last = off2;
            }
            
            currKnode[bid] = curr;
            lastKnode[bid] = last;
            offset[bid] = off;
            offset_2[bid] = off2;
            
            for (thid = 0; thid < threadsPerBlock; thid++) {
                if (knodes[curr].keys[thid] == start_val) {
                    recstart[bid] = knodes[curr].indices[thid];
                    break;
                }
            }
            
            for (thid = 0; thid < threadsPerBlock; thid++) {
                if (knodes[last].keys[thid] == end_val) {
                    reclength[bid] = knodes[last].indices[thid] - recstart[bid] + 1;
                    break;
                }
            }
        }
    }
#else
    int bid, i, thid;
    
    for (bid = 0; bid < count; bid++) {
        long curr = currKnode[bid];
        long last = lastKnode[bid];
        long off = offset[bid];
        long off2 = offset_2[bid];
        int start_val = start[bid];
        int end_val = end[bid];
        
        for (i = 0; i < maxheight; i++) {
            long next_offset = off;
            long next_offset2 = off2;
            
            for (thid = 0; thid < threadsPerBlock; thid++) {
                int lower_key = knodes[curr].keys[thid];
                int upper_key = knodes[curr].keys[thid + 1];
                
                if (lower_key <= start_val && upper_key > start_val) {
                    long idx = knodes[curr].indices[thid];
                    if (idx < knodes_elem) {
                        next_offset = idx;
                    }
                }
            }
            
            for (thid = 0; thid < threadsPerBlock; thid++) {
                int lower_key = knodes[last].keys[thid];
                int upper_key = knodes[last].keys[thid + 1];
                
                if (lower_key <= end_val && upper_key > end_val) {
                    long idx = knodes[last].indices[thid];
                    if (idx < knodes_elem) {
                        next_offset2 = idx;
                    }
                }
            }
            
            off = next_offset;
            off2 = next_offset2;
            curr = off;
            last = off2;
        }
        
        currKnode[bid] = curr;
        lastKnode[bid] = last;
        offset[bid] = off;
        offset_2[bid] = off2;
        
        for (thid = 0; thid < threadsPerBlock; thid++) {
            if (knodes[curr].keys[thid] == start_val) {
                recstart[bid] = knodes[curr].indices[thid];
                break;
            }
        }
        
        for (thid = 0; thid < threadsPerBlock; thid++) {
            if (knodes[last].keys[thid] == end_val) {
                reclength[bid] = knodes[last].indices[thid] - recstart[bid] + 1;
                break;
            }
        }
    }
#endif

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
