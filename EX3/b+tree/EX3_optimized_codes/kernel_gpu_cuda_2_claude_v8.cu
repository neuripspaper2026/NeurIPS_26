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
    __shared__ long s_lastKnode;
    __shared__ long s_offset;
    __shared__ long s_offset_2;
    __shared__ int s_start;
    __shared__ int s_end;
    
    // Initialize shared memory on first thread
    if (thid == 0) {
        s_currKnode = currKnodeD[bid];
        s_lastKnode = lastKnodeD[bid];
        s_offset = offsetD[bid];
        s_offset_2 = offset_2D[bid];
        s_start = startD[bid];
        s_end = endD[bid];
    }
    __syncthreads();

    // Process tree levels
    int i;
    for (i = 0; i < height; i++) {

        // Cache current knode pointers to reduce global memory access
        long curr_knode_idx = s_currKnode;
        long last_knode_idx = s_lastKnode;
        
        // Coalesced read: load keys and indices into registers
        int curr_key_low = knodesD[curr_knode_idx].keys[thid];
        int curr_key_high = knodesD[curr_knode_idx].keys[thid + 1];
        long curr_index_val = knodesD[curr_knode_idx].indices[thid];
        
        int last_key_low = knodesD[last_knode_idx].keys[thid];
        int last_key_high = knodesD[last_knode_idx].keys[thid + 1];
        long last_index_val = knodesD[last_knode_idx].indices[thid];

        if ((curr_key_low <= s_start) && (curr_key_high > s_start)) {
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
        if ((last_key_low <= s_end) && (last_key_high > s_end)) {
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

    // Write back final values to global memory
    if (thid == 0) {
        currKnodeD[bid] = s_currKnode;
        lastKnodeD[bid] = s_lastKnode;
        offsetD[bid] = s_offset;
        offset_2D[bid] = s_offset_2;
    }
    __syncthreads();

    // Cache final knode pointers
    long final_curr_knode = s_currKnode;
    long final_last_knode = s_lastKnode;
    
    // Coalesced read for final key comparisons
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
