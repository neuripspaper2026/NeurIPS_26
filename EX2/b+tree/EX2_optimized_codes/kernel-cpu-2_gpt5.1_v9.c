#include <stdlib.h> // (in directory known to compiler)
#include <stdio.h> // (in directory known to compiler)                  needed by printf
#ifdef _OPENMP
#include <omp.h>
#endif

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

// process number of queries
#ifdef _OPENMP
#pragma omp parallel for private(bid, i, thid) schedule(static)
#endif
    for (bid = 0; bid < count; bid++) {

        int s_key = start[bid];
        int e_key = end[bid];

        long curr = currKnode[bid];
        long last = lastKnode[bid];
        long off = offset[bid];
        long off2 = offset_2[bid];

        // process levels of the tree
        for (i = 0; i < maxheight; i++) {

            knode *kn_curr = &knodes[curr];
            knode *kn_last = &knodes[last];

            // process all leaves at each level
            for (thid = 0; thid < threadsPerBlock; thid++) {

                int k0_curr = kn_curr->keys[thid];
                int k1_curr = kn_curr->keys[thid + 1];

                if (k0_curr <= s_key && k1_curr > s_key) {

                    long idx = kn_curr->indices[thid];
                    if (idx < knodes_elem) {
                        off = idx;
                    }
                }

                int k0_last = kn_last->keys[thid];
                int k1_last = kn_last->keys[thid + 1];

                if (k0_last <= e_key && k1_last > e_key) {

                    long idx2 = kn_last->indices[thid];
                    if (idx2 < knodes_elem) {
                        off2 = idx2;
                    }
                }
            }

            // set for next tree level
            curr = off;
            last = off2;
        }

        currKnode[bid] = curr;
        lastKnode[bid] = last;
        offset[bid] = off;
        offset_2[bid] = off2;

        knode *kn_curr_final = &knodes[curr];
        knode *kn_last_final = &knodes[last];

        // process leaves - find start record
        for (thid = 0; thid < threadsPerBlock; thid++) {

            if (kn_curr_final->keys[thid] == s_key) {
                recstart[bid] = kn_curr_final->indices[thid];
            }
        }

        // process leaves - find end record / length
        int rs = recstart[bid];
        for (thid = 0; thid < threadsPerBlock; thid++) {

            if (kn_last_final->keys[thid] == e_key) {
                reclength[bid] =
                    kn_last_final->indices[thid] - rs + 1;
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
