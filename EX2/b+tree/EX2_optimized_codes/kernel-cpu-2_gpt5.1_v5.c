#include <stdlib.h> // (in directory known to compiler)
#include <stdio.h>  // (in directory known to compiler)                  needed by printf
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
    time0 = get_time();

    int threadsPerBlock;
    threadsPerBlock = order < 1024 ? order : 1024;

    time1 = get_time();

    // process number of queries
    // Parallelize across independent queries (bid)
    #pragma omp parallel for default(none) shared(count, maxheight, threadsPerBlock, knodes, knodes_elem, currKnode, lastKnode, offset, offset_2, start, end, recstart, reclength) schedule(static)
    for (int bid = 0; bid < count; bid++) {

        long localCurr     = currKnode[bid];
        long localLast     = lastKnode[bid];
        long localOffset   = offset[bid];
        long localOffset2  = offset_2[bid];
        const int sKey     = start[bid];
        const int eKey     = end[bid];
        const int tpb      = threadsPerBlock;
        const long levels  = maxheight;

        // process levels of the tree
        for (long level = 0; level < levels; level++) {

            // process all leaves at each level
            #pragma omp simd
            for (int thid = 0; thid < tpb; thid++) {

                const int *currKeys = knodes[localCurr].keys;
                const int *lastKeys = knodes[localLast].keys;

                const int k0s = currKeys[thid];
                const int k1s = currKeys[thid + 1];

                if (k0s <= sKey && k1s > sKey) {
                    const long idx = knodes[localCurr].indices[thid];
                    if (idx < knodes_elem) {
                        localOffset = idx;
                    }
                }

                const int k0e = lastKeys[thid];
                const int k1e = lastKeys[thid + 1];

                if (k0e <= eKey && k1e > eKey) {
                    const long idx2 = knodes[localLast].indices[thid];
                    if (idx2 < knodes_elem) {
                        localOffset2 = idx2;
                    }
                }
            }

            // set for next tree level
            localCurr = localOffset;
            localLast = localOffset2;
        }

        // process leaves - find start record index
        int localRecStart = recstart[bid];

        #pragma omp simd
        for (int thid = 0; thid < tpb; thid++) {

            if (knodes[localCurr].keys[thid] == sKey) {
                localRecStart = knodes[localCurr].indices[thid];
            }
        }

        // process leaves - find end record index and length
        int localRecLen = reclength[bid];

        #pragma omp simd
        for (int thid = 0; thid < tpb; thid++) {

            if (knodes[localLast].keys[thid] == eKey) {
                localRecLen = knodes[localLast].indices[thid] - localRecStart + 1;
            }
        }

        // write back results for this query
        currKnode[bid]  = localCurr;
        lastKnode[bid]  = localLast;
        offset[bid]     = localOffset;
        offset_2[bid]   = localOffset2;
        recstart[bid]   = localRecStart;
        reclength[bid]  = localRecLen;
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
