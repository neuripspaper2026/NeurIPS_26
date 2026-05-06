#include <cuda.h>
#include <cuda_runtime.h>

__global__ void findRangeK(long height,

                           knode *knodesD, long knodes_elem,

                           long *currKnodeD, long *offsetD, long *lastKnodeD,
                           long *offset_2D, int *startD, int *endD,
                           int *RecstartD, int *ReclenD) {

    // private thread IDs
    const int thid = threadIdx.x;
    const int bid  = blockIdx.x;

    // cache query range bounds per block
    const int startKey = startD[bid];
    const int endKey   = endD[bid];

    // keep traversal indices in registers
    long currNodeIndex  = currKnodeD[bid];
    long lastNodeIndex  = lastKnodeD[bid];
    long offsetNodeIdx  = offsetD[bid];
    long offset2NodeIdx = offset_2D[bid];

    // process tree levels
    for (long level = 0; level < height; ++level) {

        knode *currNode = &knodesD[currNodeIndex];
        knode *lastNode = &knodesD[lastNodeIndex];

        // load keys into registers
        int currKeyL  = currNode->keys[thid];
        int currKeyR  = currNode->keys[thid + 1];
        int lastKeyL  = lastNode->keys[thid];
        int lastKeyR  = lastNode->keys[thid + 1];

        // locate child for startKey
        if ((currKeyL <= startKey) && (currKeyR > startKey)) {
            long childIndex = currNode->indices[thid];
            // bounds check to avoid invalid access
            if (childIndex < knodes_elem) {
                offsetNodeIdx = childIndex;
            }
        }

        // locate child for endKey
        if ((lastKeyL <= endKey) && (lastKeyR > endKey)) {
            long childIndex2 = lastNode->indices[thid];
            // bounds check to avoid invalid access
            if (childIndex2 < knodes_elem) {
                offset2NodeIdx = childIndex2;
            }
        }

        __syncthreads();

        // set for next tree level
        if (thid == 0) {
            currNodeIndex = offsetNodeIdx;
            lastNodeIndex = offset2NodeIdx;
        }

        __syncthreads();
    }

    // write back final node indices for consistency with original behavior
    if (thid == 0) {
        currKnodeD[bid]  = currNodeIndex;
        lastKnodeD[bid]  = lastNodeIndex;
        offsetD[bid]     = offsetNodeIdx;
        offset_2D[bid]   = offset2NodeIdx;
    }

    __syncthreads();

    knode *startNode = &knodesD[currNodeIndex];
    knode *endNode   = &knodesD[lastNodeIndex];

    int sKey = startNode->keys[thid];

    // Find the index of the starting record
    if (sKey == startKey) {
        RecstartD[bid] = startNode->indices[thid];
    }

    __syncthreads();

    int eKey = endNode->keys[thid];

    // Find the index of the ending record
    if (eKey == endKey) {
        ReclenD[bid] =
            endNode->indices[thid] - RecstartD[bid] + 1;
    }
}
