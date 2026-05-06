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
    // Parallelize over independent queries (bid)
    #pragma omp parallel for default(none) shared(count, maxheight, threadsPerBlock, knodes, currKnode, offset, keys, knodes_elem, records, ans) schedule(static)
    for (int bid = 0; bid < count; bid++) {

        long localCurr = currKnode[bid];
        long localOff  = offset[bid];
        const int key  = keys[bid];

        // process levels of the tree
        for (long level = 0; level < maxheight; level++) {

            int nextOffsetFound = 0;

            // process all leaves at each level
            #pragma omp simd reduction(|:nextOffsetFound)
            for (int thid = 0; thid < threadsPerBlock; thid++) {

                const int k0 = knodes[localCurr].keys[thid];
                const int k1 = knodes[localCurr].keys[thid + 1];

                if (k0 <= key && k1 > key) {
                    const long idx = knodes[localOff].indices[thid];
                    if (idx < knodes_elem) {
                        localOff = idx;
                        nextOffsetFound = 1;
                    }
                }
            }

            // set for next tree level (even if not found, this matches original logic)
            localCurr = localOff;
        }

        // search final node for exact key match
        int foundValue = 0;
        int resultValue = 0;

        #pragma omp simd
        for (int thid = 0; thid < threadsPerBlock; thid++) {

            if (knodes[localCurr].keys[thid] == key && !foundValue) {
                const int idx = knodes[localCurr].indices[thid];
                resultValue = records[idx].value;
                foundValue  = 1;
            }
        }

        // write back results
        currKnode[bid] = localCurr;
        offset[bid]    = localOff;
        if (foundValue) {
            ans[bid].value = resultValue;
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
