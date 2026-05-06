#include <cuda.h>
#include <cuda_runtime.h>

__global__ void findK(long height, knode *knodesD, long knodes_elem,
                      record *recordsD,

                      long *currKnodeD, long *offsetD, int *keysD,
                      record *ansD) {

    // private thread IDs
    const int thid = threadIdx.x;
    const int bid = blockIdx.x;

    // cache frequently used values
    const int key = keysD[bid];

    // Assume node fanout = blockDim.x (number of keys per node),
    // prefetch node pointer once per iteration into a register.
    long currNodeIndex = currKnodeD[bid];
    long offsetNodeIndex = offsetD[bid];

    // process tree levels
    for (long level = 0; level < height; ++level) {

        knode *currNode = &knodesD[currNodeIndex];
        knode *offsetNode = &knodesD[offsetNodeIndex];

        // load keys and indices into registers to improve cache locality
        int key_left  = currNode->keys[thid];
        int key_right = currNode->keys[thid + 1];

        // if value is between the two keys
        if (key_left <= key && key_right > key) {
            long childIndex = offsetNode->indices[thid];

            // bounds check to avoid invalid access
            if (childIndex < knodes_elem) {
                offsetNodeIndex = childIndex;
            }
        }

        __syncthreads();

        // set for next tree level
        if (thid == 0) {
            currNodeIndex = offsetNodeIndex;
        }

        __syncthreads();
    }

    // write back the final node indices for consistency with original behavior
    if (thid == 0) {
        currKnodeD[bid] = currNodeIndex;
        offsetD[bid]    = offsetNodeIndex;
    }

    __syncthreads();

    // At this point, we have a candidate leaf node which may contain
    // the target record. Check each key to hopefully find the record.
    knode *leafNode = &knodesD[currNodeIndex];
    int leafKey = leafNode->keys[thid];

    if (leafKey == key) {
        long recIndex = leafNode->indices[thid];
        ansD[bid].value = recordsD[recIndex].value;
    }
}
