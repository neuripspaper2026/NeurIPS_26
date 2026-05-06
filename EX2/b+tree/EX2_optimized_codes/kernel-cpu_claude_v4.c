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

#ifdef _OPENMP
    #pragma omp parallel
    {
        int bid, i, thid;
        
        #pragma omp for schedule(dynamic, 1)
        for (bid = 0; bid < count; bid++) {
            long curr = currKnode[bid];
            long off = offset[bid];
            
            for (i = 0; i < maxheight; i++) {
                int found = 0;
                long next_offset = off;
                
                for (thid = 0; thid < threadsPerBlock; thid++) {
                    int key_bid = keys[bid];
                    int key_curr = knodes[curr].keys[thid];
                    int key_next = knodes[curr].keys[thid + 1];
                    
                    if (key_curr <= key_bid && key_next > key_bid) {
                        long idx = knodes[off].indices[thid];
                        if (idx < knodes_elem) {
                            next_offset = idx;
                            found = 1;
                            break;
                        }
                    }
                }
                
                off = next_offset;
                curr = off;
            }
            
            currKnode[bid] = curr;
            
            for (thid = 0; thid < threadsPerBlock; thid++) {
                if (knodes[curr].keys[thid] == keys[bid]) {
                    ans[bid].value = records[knodes[curr].indices[thid]].value;
                    break;
                }
            }
        }
    }
#else
    int bid, i, thid;
    
    for (bid = 0; bid < count; bid++) {
        long curr = currKnode[bid];
        long off = offset[bid];
        
        for (i = 0; i < maxheight; i++) {
            long next_offset = off;
            
            for (thid = 0; thid < threadsPerBlock; thid++) {
                int key_bid = keys[bid];
                int key_curr = knodes[curr].keys[thid];
                int key_next = knodes[curr].keys[thid + 1];
                
                if (key_curr <= key_bid && key_next > key_bid) {
                    long idx = knodes[off].indices[thid];
                    if (idx < knodes_elem) {
                        next_offset = idx;
                        break;
                    }
                }
            }
            
            off = next_offset;
            curr = off;
        }
        
        currKnode[bid] = curr;
        
        for (thid = 0; thid < threadsPerBlock; thid++) {
            if (knodes[curr].keys[thid] == keys[bid]) {
                ans[bid].value = records[knodes[curr].indices[thid]].value;
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
