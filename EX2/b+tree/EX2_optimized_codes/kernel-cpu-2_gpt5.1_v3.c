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

    time0 = get_time();

    int threadsPerBlock;
    threadsPerBlock = order < 1024 ? order : 1024;

    time1 = get_time();

    // process number of queries
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(knodes, knodes_elem, order, maxheight, count, currKnode, offset, lastKnode, offset_2, start, end, recstart, reclength, threadsPerBlock) schedule(static)
#endif
    for (int bid = 0; bid < count; bid++) {

        // local copies to reduce repeated memory accesses
        long curr = currKnode[bid];
        long last = lastKnode[bid];
        long off = offset[bid];
        long off2 = offset_2[bid];
        const int sKey = start[bid];
        const int eKey = end[bid];

        // process levels of the tree
        for (long i = 0; i < maxheight; i++) {

            // search child for start key
            int startChild = -1;
#pragma omp simd reduction(max : startChild)
            for (int thid = 0; thid < threadsPerBlock; thid++) {
                const int k0 = knodes[curr].keys[thid];
                const int k1 = knodes[curr].keys[thid + 1];

                if (k0 <= sKey && k1 > sKey) {
                    if (knodes[curr].indices[thid] < knodes_elem) {
                        if (thid > startChild) {
                            startChild = thid;
                        }
                    }
                }
            }

            if (startChild >= 0) {
                off = knodes[curr].indices[startChild];
            }

            // search child for end key
            int endChild = -1;
#pragma omp simd reduction(max : endChild)
            for (int thid = 0; thid < threadsPerBlock; thid++) {
                const int k0 = knodes[last].keys[thid];
                const int k1 = knodes[last].keys[thid + 1];

                if (k0 <= eKey && k1 > eKey) {
                    if (knodes[last].indices[thid] < knodes_elem) {
                        if (thid > endChild) {
                            endChild = thid;
                        }
                    }
                }
            }

            if (endChild >= 0) {
                off2 = knodes[last].indices[endChild];
            }

            // set for next tree level
            curr = off;
            last = off2;
        }

        // write back updated node positions
        currKnode[bid] = curr;
        lastKnode[bid] = last;
        offset[bid] = off;
        offset_2[bid] = off2;

        // process leaves - find starting record index
        int recStartLocal = recstart[bid];
        int matchStart = -1;
#pragma omp simd reduction(max : matchStart)
        for (int thid = 0; thid < threadsPerBlock; thid++) {
            if (knodes[curr].keys[thid] == sKey) {
                if (thid > matchStart) {
                    matchStart = thid;
                }
            }
        }
        if (matchStart >= 0) {
            recStartLocal = knodes[curr].indices[matchStart];
        }
        recstart[bid] = recStartLocal;

        // process leaves - find ending record and range length
        int matchEnd = -1;
#pragma omp simd reduction(max : matchEnd)
        for (int thid = 0; thid < threadsPerBlock; thid++) {
            if (knodes[last].keys[thid] == eKey) {
                if (thid > matchEnd) {
                    matchEnd = thid;
                }
            }
        }
        if (matchEnd >= 0) {
            reclength[bid] =
                knodes[last].indices[matchEnd] - recStartLocal + 1;
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
