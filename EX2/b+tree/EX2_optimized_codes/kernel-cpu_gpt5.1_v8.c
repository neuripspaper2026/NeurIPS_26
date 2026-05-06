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
    // Parallelize outermost loop over queries; each iteration works on distinct
    // indices (bid) and thus is independent.
    #ifdef _OPENMP
    #pragma omp parallel for default(none) shared(records, knodes, knodes_elem, order, maxheight, count, currKnode, offset, keys, ans, threadsPerBlock) schedule(static)
    #endif
    for (int bid = 0; bid < count; bid++) {

        long curr = currKnode[bid];
        long off  = offset[bid];
        const int key = keys[bid];
        const int tpb = threadsPerBlock;

        // process levels of the tree
        for (long level = 0; level < maxheight; level++) {

            long next_off = off;

            // process all leaves at each level
            // a simple linear search over the node's keys
            // find interval [keys[thid], keys[thid+1]) containing key
            for (int thid = 0; thid < tpb; thid++) {

                const knode *curr_node   = &knodes[curr];
                const long  left_key     = curr_node->keys[thid];
                const long  right_key    = curr_node->keys[thid + 1];

                if (left_key <= key && right_key > key) {
                    const knode *off_node = &knodes[off];
                    const long   idx      = off_node->indices[thid];
                    if (idx < knodes_elem) {
                        next_off = idx;
                    }
                    // Only one interval should match; we could break,
                    // but retain full loop structure for safety/consistency.
                }
            }

            // set for next tree level
            curr = next_off;
            off  = next_off;
        }

        currKnode[bid] = curr;
        offset[bid]    = off;

        {
            const knode *curr_node = &knodes[curr];
            // final exact match at leaf
            for (int thid = 0; thid < tpb; thid++) {

                if (curr_node->keys[thid] == key) {
                    const long rec_idx = curr_node->indices[thid];
                    ans[bid].value = records[rec_idx].value;
                }
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
