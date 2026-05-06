#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {

// Assuming definitions of knode and record are provided elsewhere
// and .empty_header_placeholder is included in another translation unit if needed.

__global__ void findK(long height, knode *knodesD, long knodes_elem,
                      record *recordsD,
                      long *currKnodeD, long *offsetD, int *keysD,
                      record *ansD) {

    // private thread IDs
    const int thid = threadIdx.x;
    const int bid  = blockIdx.x;

    // Cache frequently used per-block scalars
    int   key      = 0;
    long  currNode = 0;
    long  nextNode = 0;

    // Load per-block key and starting node once, from thread 0
    if (thid == 0) {
        key      = keysD[bid];
        currNode = currKnodeD[bid];
        nextNode = offsetD[bid];
    }

    // Broadcast and keep them in registers for all threads in the block
    key      = __shfl_sync(0xffffffff, key, 0);
    currNode = __shfl_sync(0xffffffff, currNode, 0);
    nextNode = __shfl_sync(0xffffffff, nextNode, 0);

    // processtree levels
    for (long level = 0; level < height; level++) {

        // Load node index into a register
        long nodeIdx = currNode;

        // Access node structure once per thread; rely on compiler to keep in registers
        knode *nodePtr = &knodesD[nodeIdx];

        int keyLeft  = nodePtr->keys[thid];
        int keyRight = nodePtr->keys[thid + 1];

        // if value is between the two keys
        if (keyLeft <= key && keyRight > key) {

            // use cached nextNode for this level
            long childNodeIdx = nextNode;
            knode *childNodePtr = &knodesD[childNodeIdx];

            long childIndex = childNodePtr->indices[thid];

            // bounds check as in original code
            if (childIndex < knodes_elem) {
                // update next node in register
                nextNode = childIndex;
            }
        }

        __syncthreads();

        // set for next tree level
        if (thid == 0) {
            currNode = nextNode;
        }

        // Broadcast updated currNode and nextNode to all threads
        currNode = __shfl_sync(0xffffffff, currNode, 0);
        nextNode = __shfl_sync(0xffffffff, nextNode, 0);

        __syncthreads();
    }

    // Write back the final node indices to global memory from thread 0
    if (thid == 0) {
        currKnodeD[bid] = currNode;
        offsetD[bid]    = nextNode;
    }

    __syncthreads();

    // At this point, we have a candidate leaf node which may contain
    // the target record.  Check each key to hopefully find the record
    long finalNodeIdx = currNode;
    knode *finalNodePtr = &knodesD[finalNodeIdx];

    if (finalNodePtr->keys[thid] == key) {
        long recIdx = finalNodePtr->indices[thid];
        ansD[bid].value = recordsD[recIdx].value;
    }
}

} // extern "C"
