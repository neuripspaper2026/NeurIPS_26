#include "empty_header_placeholder"

__global__ void findRangeK(long height,

                           knode *knodesD, long knodes_elem,

                           long *currKnodeD, long *offsetD, long *lastKnodeD,
                           long *offset_2D, int *startD, int *endD,
                           int *RecstartD, int *ReclenD) {

    // private thread IDs
    const int thid = threadIdx.x;
    const int bid  = blockIdx.x;

    // cache frequently used scalars in registers
    int  start_key = startD[bid];
    int  end_key   = endD[bid];
    long curr      = currKnodeD[bid];
    long last      = lastKnodeD[bid];
    long off       = offsetD[bid];
    long off2      = offset_2D[bid];

    // process tree levels
    for (long level = 0; level < height; level++) {

        // load node pointers once per thread
        knode *currNodePtr = &knodesD[curr];
        knode *lastNodePtr = &knodesD[last];

        int currKey      = currNodePtr->keys[thid];
        int currKeyNext  = currNodePtr->keys[thid + 1];
        int lastKey      = lastNodePtr->keys[thid];
        int lastKeyNext  = lastNodePtr->keys[thid + 1];

        // locate child for start_key
        if (currKey <= start_key && currKeyNext > start_key) {
            long child = currNodePtr->indices[thid];
            if (child < knodes_elem) {
                off = child;
                offsetD[bid] = child;
            }
        }

        // locate child for end_key
        if (lastKey <= end_key && lastKeyNext > end_key) {
            long child2 = lastNodePtr->indices[thid];
            if (child2 < knodes_elem) {
                off2 = child2;
                offset_2D[bid] = child2;
            }
        }

        __syncthreads();

        // set for next tree level
        if (thid == 0) {
            curr = off;
            last = off2;
            currKnodeD[bid] = curr;
            lastKnodeD[bid] = last;
        }
        __syncthreads();
    }

    // Find the index of the starting record
    knode *finalCurrNodePtr = &knodesD[currKnodeD[bid]];
    int    finalCurrKey     = finalCurrNodePtr->keys[thid];

    if (finalCurrKey == start_key) {
        RecstartD[bid] = finalCurrNodePtr->indices[thid];
    }
    __syncthreads();

    // Find the index of the ending record
    knode *finalLastNodePtr = &knodesD[lastKnodeD[bid]];
    int    finalLastKey     = finalLastNodePtr->keys[thid];

    if (finalLastKey == end_key) {
        ReclenD[bid] =
            finalLastNodePtr->indices[thid] - RecstartD[bid] + 1;
    }
}
