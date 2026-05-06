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

    // cache per-query scalars in registers
    const int  startKey = startD[bid];
    const int  endKey   = endD[bid];

    // cache current and last knode indices in registers
    long currKnode  = currKnodeD[bid];
    long lastKnode  = lastKnodeD[bid];

    // traverse tree levels
    for (long level = 0; level < height; ++level) {

        // load node pointers once per level
        knode *currNodePtr = &knodesD[currKnode];
        knode *lastNodePtr = &knodesD[lastKnode];

        // cache keys for current and last nodes in registers
        const int currKeyLeft   = currNodePtr->keys[thid];
        const int currKeyRight  = currNodePtr->keys[thid + 1];
        const int lastKeyLeft   = lastNodePtr->keys[thid];
        const int lastKeyRight  = lastNodePtr->keys[thid + 1];

        // check range for start key
        if (currKeyLeft <= startKey && currKeyRight > startKey) {
            const long childIdx = currNodePtr->indices[thid];
            // bounds guard
            if (childIdx < knodes_elem) {
                offsetD[bid] = childIdx;
            }
        }

        // check range for end key
        if (lastKeyLeft <= endKey && lastKeyRight > endKey) {
            const long childIdx2 = lastNodePtr->indices[thid];
            // bounds guard
            if (childIdx2 < knodes_elem) {
                offset_2D[bid] = childIdx2;
            }
        }

        __syncthreads();

        // set for next tree level
        if (thid == 0) {
            currKnode = offsetD[bid];
            lastKnode = offset_2D[bid];

            currKnodeD[bid] = currKnode;
            lastKnodeD[bid] = lastKnode;
        }
        __syncthreads();
    }

    // Find the index of the starting record
    {
        knode *leafStartPtr = &knodesD[currKnode];
        const int leafKeyStart = leafStartPtr->keys[thid];

        if (leafKeyStart == startKey) {
            RecstartD[bid] = leafStartPtr->indices[thid];
        }
    }
    __syncthreads();

    // Find the index of the ending record
    {
        knode *leafEndPtr = &knodesD[lastKnode];
        const int leafKeyEnd = leafEndPtr->keys[thid];

        if (leafKeyEnd == endKey) {
            ReclenD[bid] = leafEndPtr->indices[thid] - RecstartD[bid] + 1;
        }
    }
}
