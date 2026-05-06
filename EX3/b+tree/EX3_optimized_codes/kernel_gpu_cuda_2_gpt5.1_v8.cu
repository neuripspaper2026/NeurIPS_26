#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {

__global__ void findRangeK(long height,

                           knode *knodesD, long knodes_elem,

                           long *currKnodeD, long *offsetD, long *lastKnodeD,
                           long *offset_2D, int *startD, int *endD,
                           int *RecstartD, int *ReclenD) {

    const int thid = threadIdx.x;
    const int bid  = blockIdx.x;

    // Per-block cached scalars
    long currNode   = 0;
    long lastNode   = 0;
    long offset     = 0;
    long offset2    = 0;
    int  startKey   = 0;
    int  endKey     = 0;

    // Load per-query state once from global memory using thread 0
    if (thid == 0) {
        currNode = currKnodeD[bid];
        lastNode = lastKnodeD[bid];
        offset   = offsetD[bid];
        offset2  = offset_2D[bid];
        startKey = startD[bid];
        endKey   = endD[bid];
    }

    // Broadcast values from lane 0 to all threads in the block
    currNode = __shfl_sync(0xffffffff, currNode, 0);
    lastNode = __shfl_sync(0xffffffff, lastNode, 0);
    offset   = __shfl_sync(0xffffffff, offset, 0);
    offset2  = __shfl_sync(0xffffffff, offset2, 0);
    startKey = __shfl_sync(0xffffffff, startKey, 0);
    endKey   = __shfl_sync(0xffffffff, endKey, 0);

    // Tree traversal for the given height
    for (long level = 0; level < height; level++) {

        // Cache node pointers in registers
        knode *currNodePtr = &knodesD[currNode];
        knode *lastNodePtr = &knodesD[lastNode];

        int cKeyLeft  = currNodePtr->keys[thid];
        int cKeyRight = currNodePtr->keys[thid + 1];

        if (cKeyLeft <= startKey && cKeyRight > startKey) {
            long childIdx = currNodePtr->indices[thid];
            if (childIdx < knodes_elem) {
                offset = childIdx;
            }
        }

        int lKeyLeft  = lastNodePtr->keys[thid];
        int lKeyRight = lastNodePtr->keys[thid + 1];

        if (lKeyLeft <= endKey && lKeyRight > endKey) {
            long childIdx2 = lastNodePtr->indices[thid];
            if (childIdx2 < knodes_elem) {
                offset2 = childIdx2;
            }
        }

        __syncthreads();

        // Update nodes for next level using thread 0
        if (thid == 0) {
            currNode = offset;
            lastNode = offset2;
        }

        // Broadcast updated node indices and offsets to all threads
        currNode = __shfl_sync(0xffffffff, currNode, 0);
        lastNode = __shfl_sync(0xffffffff, lastNode, 0);
        offset   = __shfl_sync(0xffffffff, offset, 0);
        offset2  = __shfl_sync(0xffffffff, offset2, 0);

        __syncthreads();
    }

    // Write back final traversal state once from thread 0
    if (thid == 0) {
        currKnodeD[bid]  = currNode;
        lastKnodeD[bid]  = lastNode;
        offsetD[bid]     = offset;
        offset_2D[bid]   = offset2;
        startD[bid]      = startKey;
        endD[bid]        = endKey;
    }

    __syncthreads();

    // Find the index of the starting record
    knode *finalStartNodePtr = &knodesD[currNode];
    if (finalStartNodePtr->keys[thid] == startKey) {
        RecstartD[bid] = finalStartNodePtr->indices[thid];
    }

    __syncthreads();

    // Find the index of the ending record
    knode *finalEndNodePtr = &knodesD[lastNode];
    if (finalEndNodePtr->keys[thid] == endKey) {
        ReclenD[bid] = finalEndNodePtr->indices[thid] - RecstartD[bid] + 1;
    }
}

} // extern "C"
