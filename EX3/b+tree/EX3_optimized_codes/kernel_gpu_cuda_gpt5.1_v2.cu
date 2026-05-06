#include "empty_header_placeholder"

__global__ void findK(long height, knode *knodesD, long knodes_elem,
                      record *recordsD,

                      long *currKnodeD, long *offsetD, int *keysD,
                      record *ansD) {

    // private thread IDs
    const int thid = threadIdx.x;
    const int bid = blockIdx.x;

    // cache frequently used scalars in registers
    int key = keysD[bid];
    long curr = currKnodeD[bid];
    long off  = offsetD[bid];

    // process tree levels
    for (long level = 0; level < height; level++) {

        // load node indices once per thread
        long nodeIdx = currKnodeD[bid];
        knode *nodePtr = &knodesD[nodeIdx];

        int myKey      = nodePtr->keys[thid];
        int myKeyNext  = nodePtr->keys[thid + 1];

        // if value is between the two keys
        if (myKey <= key && myKeyNext > key) {

            long offIdx = offsetD[bid];
            knode *offPtr = &knodesD[offIdx];

            long child = offPtr->indices[thid];

            // avoid out-of-bounds
            if (child < knodes_elem) {
                off = child;
                offsetD[bid] = child;
            }
        }
        __syncthreads();

        // set for next tree level
        if (thid == 0) {
            curr = off;
            currKnodeD[bid] = off;
        }
        __syncthreads();
    }

    // At this point, we have a candidate leaf node which may contain
    // the target record.  Check each key to hopefully find the record
    long finalNodeIdx = currKnodeD[bid];
    knode *finalNodePtr = &knodesD[finalNodeIdx];

    if (finalNodePtr->keys[thid] == key) {
        long recIdx = finalNodePtr->indices[thid];
        ansD[bid].value = recordsD[recIdx].value;
    }
}
