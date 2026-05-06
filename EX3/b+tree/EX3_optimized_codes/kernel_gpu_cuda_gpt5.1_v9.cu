#include <cuda.h>
#include <cuda_runtime.h>

__global__ void findK(long height, knode *knodesD, long knodes_elem,
                      record *recordsD,

                      long *currKnodeD, long *offsetD, int *keysD,
                      record *ansD) {

    // private thread IDs
    const int thid = threadIdx.x;
    const int bid  = blockIdx.x;

    // cache frequently used key in a register
    const int searchKey = keysD[bid];

    // cache current knode in a register
    long currKnode = currKnodeD[bid];

    // process tree levels
    for (long level = 0; level < height; ++level) {

        // load knode pointer once per level
        knode *nodePtr = &knodesD[currKnode];

        // load keys into registers to avoid repeated global memory access
        const int keyLeft  = nodePtr->keys[thid];
        const int keyRight = nodePtr->keys[thid + 1];

        // if value is between the two keys
        if (keyLeft <= searchKey && keyRight > searchKey) {
            long off = offsetD[bid];
            knode *offNodePtr = &knodesD[off];

            const long idx = offNodePtr->indices[thid];

            // guard against out-of-bounds
            if (idx < knodes_elem) {
                offsetD[bid] = idx;
            }
        }

        __syncthreads();

        // set for next tree level
        if (thid == 0) {
            currKnode = offsetD[bid];
            currKnodeD[bid] = currKnode;
        }

        __syncthreads();
    }

    // At this point, we have a candidate leaf node which may contain
    // the target record.  Check each key to hopefully find the record
    knode *leafPtr = &knodesD[currKnode];
    const int leafKey = leafPtr->keys[thid];

    if (leafKey == searchKey) {
        const long recIdx = leafPtr->indices[thid];
        ansD[bid].value = recordsD[recIdx].value;
    }
}
