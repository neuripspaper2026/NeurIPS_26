__global__ void findRangeK(long height,

                           knode *knodesD, long knodes_elem,

                           long *currKnodeD, long *offsetD, long *lastKnodeD,
                           long *offset_2D, int *startD, int *endD,
                           int *RecstartD, int *ReclenD) {

    // private thread IDs
    const int thid = threadIdx.x;
    const int bid  = blockIdx.x;

    // cache frequently used per-query values in registers
    int  start    = startD[bid];
    int  end      = endD[bid];
    long curr     = currKnodeD[bid];
    long last     = lastKnodeD[bid];
    long offset   = offsetD[bid];
    long offset_2 = offset_2D[bid];

    // traverse the tree for the specified height
    for (long level = 0; level < height; ++level) {

        // cache node indices locally
        const long currIdx = curr;
        const long lastIdx = last;

        // load keys for both current and last nodes into registers
        const int keyCurrL = knodesD[currIdx].keys[thid];
        const int keyCurrR = knodesD[currIdx].keys[thid + 1];
        const int keyLastL = knodesD[lastIdx].keys[thid];
        const int keyLastR = knodesD[lastIdx].keys[thid + 1];

        // load child indices for both nodes
        const long childCurr = knodesD[currIdx].indices[thid];
        const long childLast = knodesD[lastIdx].indices[thid];

        // search in current node for start key
        if (keyCurrL <= start && keyCurrR > start) {
            // this conditional statement is inserted to avoid crash due to bug
            // in original code
            // "offset[bid]" calculated below that later addresses part of
            // knodes goes outside of its bounds cause segmentation fault
            // more specifically, values saved into knodes->indices in the main
            // function are out of bounds of knodes that they address
            if (childCurr < knodes_elem) {
                offset = childCurr;
            }
        }

        // search in last node for end key
        if (keyLastL <= end && keyLastR > end) {
            // this conditional statement is inserted to avoid crash due to bug
            // in original code
            // "offset_2[bid]" calculated below that later addresses part of
            // knodes goes outside of its bounds cause segmentation fault
            // more specifically, values saved into knodes->indices in the main
            // function are out of bounds of knodes that they address
            if (childLast < knodes_elem) {
                offset_2 = childLast;
            }
        }

        __syncthreads();

        // set for next tree level (single-thread commit to global state)
        if (thid == 0) {
            curr = offset;
            last = offset_2;

            currKnodeD[bid]  = curr;
            lastKnodeD[bid]  = last;
            offsetD[bid]     = offset;
            offset_2D[bid]   = offset_2;
        }
        __syncthreads();
    }

    // reload final leaf nodes into registers for clarity
    const long leafStartIdx = currKnodeD[bid];
    const long leafEndIdx   = lastKnodeD[bid];

    // Find the index of the starting record
    if (knodesD[leafStartIdx].keys[thid] == start) {
        RecstartD[bid] = knodesD[leafStartIdx].indices[thid];
    }
    __syncthreads();

    // Find the index of the ending record
    if (knodesD[leafEndIdx].keys[thid] == end) {
        ReclenD[bid] =
            knodesD[leafEndIdx].indices[thid] - RecstartD[bid] + 1;
    }
}
