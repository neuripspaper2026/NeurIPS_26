__global__ void findRangeK(long height,

                           knode *knodesD, long knodes_elem,

                           long *currKnodeD, long *offsetD, long *lastKnodeD,
                           long *offset_2D, int *startD, int *endD,
                           int *RecstartD, int *ReclenD) {

    // private thread IDs
    int thid = threadIdx.x;
    int bid = blockIdx.x;

    // Shared memory for caching frequently accessed data
    __shared__ long shared_currKnode;
    __shared__ long shared_lastKnode;
    __shared__ long shared_offset;
    __shared__ long shared_offset_2;

    // Initialize shared memory
    if (thid == 0) {
        shared_currKnode = currKnodeD[bid];
        shared_lastKnode = lastKnodeD[bid];
        shared_offset = offsetD[bid];
        shared_offset_2 = offset_2D[bid];
    }
    __syncthreads();

    // Process tree levels
    for (int i = 0; i < height; i++) {

        // Check start key range
        if ((knodesD[shared_currKnode].keys[thid] <= startD[bid]) &&
            (knodesD[shared_currKnode].keys[thid + 1] > startD[bid])) {
            // Avoid out-of-bounds access
            if (knodesD[shared_currKnode].indices[thid] < knodes_elem) {
                shared_offset = knodesD[shared_currKnode].indices[thid];
            }
        }

        // Check end key range
        if ((knodesD[shared_lastKnode].keys[thid] <= endD[bid]) &&
            (knodesD[shared_lastKnode].keys[thid + 1] > endD[bid])) {
            // Avoid out-of-bounds access
            if (knodesD[shared_lastKnode].indices[thid] < knodes_elem) {
                shared_offset_2 = knodesD[shared_lastKnode].indices[thid];
            }
        }
        __syncthreads();

        // Set for next tree level
        if (thid == 0) {
            shared_currKnode = shared_offset;
            shared_lastKnode = shared_offset_2;
        }
        __syncthreads();
    }

    // Find the index of the starting record
    if (knodesD[shared_currKnode].keys[thid] == startD[bid]) {
        RecstartD[bid] = knodesD[shared_currKnode].indices[thid];
    }
    __syncthreads();

    // Find the index of the ending record
    if (knodesD[shared_lastKnode].keys[thid] == endD[bid]) {
        ReclenD[bid] =
            knodesD[shared_lastKnode].indices[thid] - RecstartD[bid] + 1;
    }
}
