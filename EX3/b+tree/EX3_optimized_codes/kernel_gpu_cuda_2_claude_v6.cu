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
    
    // Initialize shared memory once per block
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
        
        // Coalesced read of keys - threads read consecutive memory locations
        int curr_key_lower = knodesD[curr_knode_idx].keys[thid];
        int curr_key_upper = knodesD[curr_knode_idx].keys[thid + 1];
        long curr_index_val = knodesD[curr_knode_idx].indices[thid];
        
        int last_key_lower = knodesD[last_knode_idx].keys[thid];
        int last_key_upper = knodesD[last_knode_idx].keys[thid + 1];
        long last_index_val = knodesD[last_knode_idx].indices[thid];

        if ((curr_key_lower <= s_start) && (curr_key_upper > s_start)) {
            // this conditional statement is inserted to avoid crush due to but
            // in original code
            // "offset[bid]" calculated below that later addresses part of
            // knodes goes outside of its bounds cause segmentation fault
            // more specifically, values saved into knodes->indices in the main
            // function are out of bounds of knodes that they address
            if (curr_index_val < knodes_elem) {
                s_offset = curr_index_val;
            }
        }
        if ((last_key_lower <= s_end) && (last_key_upper > s_end)) {
            // this conditional statement is inserted to avoid crush due to but
            // in original code
            // "offset_2[bid]" calculated below that later addresses part of
            // knodes goes outside of its bounds cause segmentation fault
            // more specifically, values saved into knodes->indices in the main
            // function are out of bounds of knodes that they address
            if (last_index_val < knodes_elem) {
                s_offset_2 = last_index_val;
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

    // Write back to global memory once before final searches
    if (thid == 0) {
        currKnodeD[bid] = s_currKnode;
        offsetD[bid] = s_offset;
        lastKnodeD[bid] = s_lastKnode;
        offset_2D[bid] = s_offset_2;
    }

    // Find the index of the starting record
    int final_curr_key = knodesD[s_currKnode].keys[thid];
    if (final_curr_key == s_start) {
        RecstartD[bid] = knodesD[s_currKnode].indices[thid];
    }
    __syncthreads();

    // Find the index of the ending record
    int final_last_key = knodesD[s_lastKnode].keys[thid];
    if (final_last_key == s_end) {
        ReclenD[bid] =
            knodesD[s_lastKnode].indices[thid] - RecstartD[bid] + 1;
    }
}
