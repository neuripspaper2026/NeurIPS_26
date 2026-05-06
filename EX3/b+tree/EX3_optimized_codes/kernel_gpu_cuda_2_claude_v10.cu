__global__ void findRangeK(long height,

                           knode *knodesD, long knodes_elem,

                           long *currKnodeD, long *offsetD, long *lastKnodeD,
                           long *offset_2D, int *startD, int *endD,
                           int *RecstartD, int *ReclenD) {

    // private thread IDs
    int thid = threadIdx.x;
    int bid = blockIdx.x;

    // Load frequently accessed data into registers
    long currKnode = currKnodeD[bid];
    long lastKnode = lastKnodeD[bid];
    long offset = offsetD[bid];
    long offset_2 = offset_2D[bid];
    int start = startD[bid];
    int end = endD[bid];

    // ???
    int i;
    for (i = 0; i < height; i++) {

        // Load keys and indices into shared memory for coalesced access
        __shared__ int curr_keys[33];
        __shared__ long curr_indices[32];
        __shared__ int last_keys[33];
        __shared__ long last_indices[32];
        
        if (thid < 33) {
            curr_keys[thid] = knodesD[currKnode].keys[thid];
            last_keys[thid] = knodesD[lastKnode].keys[thid];
        }
        if (thid < 32) {
            curr_indices[thid] = knodesD[currKnode].indices[thid];
            last_indices[thid] = knodesD[lastKnode].indices[thid];
        }
        __syncthreads();

        if ((curr_keys[thid] <= start) &&
            (curr_keys[thid + 1] > start)) {
            // this conditional statement is inserted to avoid crush due to but
            // in original code
            // "offset[bid]" calculated below that later addresses part of
            // knodes goes outside of its bounds cause segmentation fault
            // more specifically, values saved into knodes->indices in the main
            // function are out of bounds of knodes that they address
            long next_offset = curr_indices[thid];
            if (next_offset < knodes_elem) {
                offset = next_offset;
            }
        }
        if ((last_keys[thid] <= end) &&
            (last_keys[thid + 1] > end)) {
            // this conditional statement is inserted to avoid crush due to but
            // in original code
            // "offset_2[bid]" calculated below that later addresses part of
            // knodes goes outside of its bounds cause segmentation fault
            // more specifically, values saved into knodes->indices in the main
            // function are out of bounds of knodes that they address
            long next_offset_2 = last_indices[thid];
            if (next_offset_2 < knodes_elem) {
                offset_2 = next_offset_2;
            }
        }
        __syncthreads();

        // set for next tree level
        if (thid == 0) {
            currKnode = offset;
            lastKnode = offset_2;
        }
        __syncthreads();
        
        // Broadcast updated values to all threads using warp shuffle
        currKnode = __shfl_sync(0xFFFFFFFF, currKnode, 0);
        lastKnode = __shfl_sync(0xFFFFFFFF, lastKnode, 0);
    }

    // Load final node keys and indices into shared memory
    __shared__ int final_curr_keys[33];
    __shared__ long final_curr_indices[32];
    __shared__ int final_last_keys[33];
    __shared__ long final_last_indices[32];
    
    if (thid < 33) {
        final_curr_keys[thid] = knodesD[currKnode].keys[thid];
        final_last_keys[thid] = knodesD[lastKnode].keys[thid];
    }
    if (thid < 32) {
        final_curr_indices[thid] = knodesD[currKnode].indices[thid];
        final_last_indices[thid] = knodesD[lastKnode].indices[thid];
    }
    __syncthreads();

    // Find the index of the starting record
    if (final_curr_keys[thid] == start) {
        RecstartD[bid] = final_curr_indices[thid];
    }
    __syncthreads();

    // Find the index of the ending record
    if (final_last_keys[thid] == end) {
        ReclenD[bid] = final_last_indices[thid] - RecstartD[bid] + 1;
    }
    
    // Write back results to global memory
    if (thid == 0) {
        currKnodeD[bid] = currKnode;
        lastKnodeD[bid] = lastKnode;
        offsetD[bid] = offset;
        offset_2D[bid] = offset_2;
    }
}
