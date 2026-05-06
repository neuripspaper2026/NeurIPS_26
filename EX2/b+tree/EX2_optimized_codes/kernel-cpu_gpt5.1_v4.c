#include <stdlib.h> // (in directory known to compiler)			needed by malloc
#include <stdio.h> // (in directory known to compiler)			needed by printf, stderr
#ifdef _OPENMP
#include <omp.h>
#endif

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

    // process number of queries
#ifdef _OPENMP
#pragma omp parallel for default(none) schedule(static) \
    shared(count, maxheight, threadsPerBlock, knodes, currKnode, offset, keys, ans, records, knodes_elem) \
    if(count > 1)
#endif
    for (int bid = 0; bid < count; bid++) {

        long curr = currKnode[bid];
        long off  = offset[bid];
        const int key = keys[bid];

        // process levels of the tree
        for (long level = 0; level < maxheight; level++) {

            // process all leaves at each level
            for (int thid = 0; thid < threadsPerBlock; thid++) {

                // if value is between the two keys
                const int k0 = knodes[curr].keys[thid];
                const int k1 = knodes[curr].keys[thid + 1];

                if (k0 <= key && k1 > key) {

                    const long idx = knodes[off].indices[thid];
                    if (idx < knodes_elem) {
                        off = idx;
                    }
                }
            }

            // set for next tree level
            curr = off;
        }

        currKnode[bid] = curr;
        offset[bid]    = off;

        for (int thid = 0; thid < threadsPerBlock; thid++) {

            if (knodes[curr].keys[thid] == key) {
                ans[bid].value = records[knodes[curr].indices[thid]].value;
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
