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

#ifdef _OPENMP
    #pragma omp parallel
    {
        int bid;
        #pragma omp for schedule(dynamic, 1)
        for (bid = 0; bid < count; bid++) {
            long curr = currKnode[bid];
            long last = lastKnode[bid];
            long off = offset[bid];
            long off2 = offset_2[bid];

            for (i = 0; i < maxheight; i++) {
                int found_start = -1;
                int found_end = -1;

                for (int thid = 0; thid < threadsPerBlock; thid++) {
                    if (found_start < 0 &&
                        (knodes[curr].keys[thid] <= start[bid]) &&
                        (knodes[curr].keys[thid + 1] > start[bid])) {
                        found_start = thid;
                    }
                    if (found_end < 0 &&
                        (knodes[last].keys[thid] <= end[bid]) &&
                        (knodes[last].keys[thid + 1] > end[bid])) {
                        found_end = thid;
                    }
                    if (found_start >= 0 && found_end >= 0) {
                        break;
                    }
                }

                if (found_start >= 0) {
                    long next_idx = knodes[curr].indices[found_start];
                    if (next_idx < knodes_elem) {
                        off = next_idx;
                    }
                }
                if (found_end >= 0) {
                    long next_idx = knodes[last].indices[found_end];
                    if (next_idx < knodes_elem) {
                        off2 = next_idx;
                    }
                }

                curr = off;
                last = off2;
            }

            offset[bid] = off;
            offset_2[bid] = off2;
            currKnode[bid] = curr;
            lastKnode[bid] = last;

            for (int thid = 0; thid < threadsPerBlock; thid++) {
                if (knodes[curr].keys[thid] == start[bid]) {
                    recstart[bid] = knodes[curr].indices[thid];
                    break;
                }
            }

            for (int thid = 0; thid < threadsPerBlock; thid++) {
                if (knodes[last].keys[thid] == end[bid]) {
                    reclength[bid] =
                        knodes[last].indices[thid] - recstart[bid] + 1;
                    break;
                }
            }
        }
    }
#else
    for (int bid = 0; bid < count; bid++) {
        long curr = currKnode[bid];
        long last = lastKnode[bid];
        long off = offset[bid];
        long off2 = offset_2[bid];

        for (i = 0; i < maxheight; i++) {
            int found_start = -1;
            int found_end = -1;

            for (int thid = 0; thid < threadsPerBlock; thid++) {
                if (found_start < 0 &&
                    (knodes[curr].keys[thid] <= start[bid]) &&
                    (knodes[curr].keys[thid + 1] > start[bid])) {
                    found_start = thid;
                }
                if (found_end < 0 &&
                    (knodes[last].keys[thid] <= end[bid]) &&
                    (knodes[last].keys[thid + 1] > end[bid])) {
                    found_end = thid;
                }
                if (found_start >= 0 && found_end >= 0) {
                    break;
                }
            }

            if (found_start >= 0) {
                long next_idx = knodes[curr].indices[found_start];
                if (next_idx < knodes_elem) {
                    off = next_idx;
                }
            }
            if (found_end >= 0) {
                long next_idx = knodes[last].indices[found_end];
                if (next_idx < knodes_elem) {
                    off2 = next_idx;
                }
            }

            curr = off;
            last = off2;
        }

        offset[bid] = off;
        offset_2[bid] = off2;
        currKnode[bid] = curr;
        lastKnode[bid] = last;

        for (int thid = 0; thid < threadsPerBlock; thid++) {
            if (knodes[curr].keys[thid] == start[bid]) {
                recstart[bid] = knodes[curr].indices[thid];
                break;
            }
        }

        for (int thid = 0; thid < threadsPerBlock; thid++) {
            if (knodes[last].keys[thid] == end[bid]) {
                reclength[bid] =
                    knodes[last].indices[thid] - recstart[bid] + 1;
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
