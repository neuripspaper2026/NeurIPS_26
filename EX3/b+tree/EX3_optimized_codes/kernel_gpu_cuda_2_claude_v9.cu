__global__ void findRangeK(long height,

                           knode *knodesD, long knodes_elem,

                           long *currKnodeD, long *offsetD, long *lastKnodeD,
                           long *offset_2D, int *startD, int *endD,
                           int *RecstartD, int *ReclenD) {

    // private thread IDs
    int thid = threadIdx.x;
    int bid = blockIdx.x;

    // Load frequently accessed data into shared memory
    __shared__ long s_currKnode;
    __shared__ long s_offset;
    __shared__ long s_lastKnode;
    __shared__ long s_offset_2;
    __shared__ int s_start;
    __shared__ int s_end;

    // Initialize shared memory on first thread
    if (thid == 0) {
        s_currKnode = currKnodeD[bid];
        s_offset = offsetD[bid];
        s_lastKnode = lastKnodeD[bid];
        s_offset_2 = offset_2D[bid];
        s_start = startD[bid];
        s_end = endD[bid];
    }
    __syncthreads();

    // Process tree levels
    int i;
    for (i = 0; i < height; i++) {

        // Cache current knode pointers for this iteration
        long curr_knode_idx = s_currKnode;
        long last_knode_idx = s_lastKnode;

        // Coalesced read of keys and indices - each thread reads its own data
        int curr_key_lower = knodesD[curr_knode_idx].keys[thid];
        int curr_key_upper = knodesD[curr_knode_idx].keys[thid + 1];
        long curr_indices_val = knodesD[curr_knode_idx].indices[thid];

        int last_key_lower = knodesD[last_knode_idx].keys[thid];
        int last_key_upper = knodesD[last_knode_idx].keys[thid + 1];
        long last_indices_val = knodesD[last_knode_idx].indices[thid];

        // Check start condition
        if ((curr_key_lower <= s_start) && (curr_key_upper > s_start)) {
            if (curr_indices_val < knodes_elem) {
                s_offset = curr_indices_val;
            }
        }

        // Check end condition
        if ((last_key_lower <= s_end) && (last_key_upper > s_end)) {
            if (last_indices_val < knodes_elem) {
                s_offset_2 = last_indices_val;
            }
        }
        __syncthreads();

        // set for next tree level
        if (thid == 0) {
            s_currKnode = s_offset;
            s_lastKnode = s_offset_2;
        }
        __syncthreads();
    }

    // Write back final values to global memory
    if (thid == 0) {
        currKnodeD[bid] = s_currKnode;
        offsetD[bid] = s_offset;
        lastKnodeD[bid] = s_lastKnode;
        offset_2D[bid] = s_offset_2;
    }

    // Cache final knode pointers
    long final_curr_knode = s_currKnode;
    long final_last_knode = s_lastKnode;

    // Coalesced read for final search
    int final_curr_key = knodesD[final_curr_knode].keys[thid];
    long final_curr_index = knodesD[final_curr_knode].indices[thid];

    int final_last_key = knodesD[final_last_knode].keys[thid];
    long final_last_index = knodesD[final_last_knode].indices[thid];

    // Find the index of the starting record
    if (final_curr_key == s_start) {
        RecstartD[bid] = final_curr_index;
    }
    __syncthreads();

    // Find the index of the ending record
    if (final_last_key == s_end) {
        ReclenD[bid] = final_last_index - RecstartD[bid] + 1;
    }
}
