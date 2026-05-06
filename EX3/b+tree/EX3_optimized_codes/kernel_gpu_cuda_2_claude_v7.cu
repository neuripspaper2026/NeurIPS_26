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
    __shared__ int s_recstart;
    __shared__ int s_reclen;
    
    // Initialize shared memory on first thread
    if (thid == 0) {
        s_currKnode = currKnodeD[bid];
        s_offset = offsetD[bid];
        s_lastKnode = lastKnodeD[bid];
        s_offset_2 = offset_2D[bid];
        s_start = startD[bid];
        s_end = endD[bid];
        s_recstart = 0;
        s_reclen = 0;
    }
    __syncthreads();

    // ???
    int i;
    for (i = 0; i < height; i++) {

        // Cache current knode pointers for coalesced access
        long curr_knode_idx = s_currKnode;
        long last_knode_idx = s_lastKnode;
        
        // Load keys into registers for this thread
        int curr_key_lower = knodesD[curr_knode_idx].keys[thid];
        int curr_key_upper = knodesD[curr_knode_idx].keys[thid + 1];
        int last_key_lower = knodesD[last_knode_idx].keys[thid];
        int last_key_upper = knodesD[last_knode_idx].keys[thid + 1];
        int start_val = s_start;
        int end_val = s_end;

        if ((curr_key_lower <= start_val) && (curr_key_upper > start_val)) {
            // this conditional statement is inserted to avoid crush due to but
            // in original code
            // "offset[bid]" calculated below that later addresses part of
            // knodes goes outside of its bounds cause segmentation fault
            // more specifically, values saved into knodes->indices in the main
            // function are out of bounds of knodes that they address
            long next_idx = knodesD[curr_knode_idx].indices[thid];
            if (next_idx < knodes_elem) {
                s_offset = next_idx;
            }
        }
        if ((last_key_lower <= end_val) && (last_key_upper > end_val)) {
            // this conditional statement is inserted to avoid crush due to but
            // in original code
            // "offset_2[bid]" calculated below that later addresses part of
            // knodes goes outside of its bounds cause segmentation fault
            // more specifically, values saved into knodes->indices in the main
            // function are out of bounds of knodes that they address
            long next_idx_2 = knodesD[last_knode_idx].indices[thid];
            if (next_idx_2 < knodes_elem) {
                s_offset_2 = next_idx_2;
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

    // Cache final knode pointers
    long final_curr_knode_idx = s_currKnode;
    long final_last_knode_idx = s_lastKnode;
    int final_start = s_start;
    int final_end = s_end;

    // Find the index of the starting record
    if (knodesD[final_curr_knode_idx].keys[thid] == final_start) {
        s_recstart = knodesD[final_curr_knode_idx].indices[thid];
    }
    __syncthreads();

    // Find the index of the ending record
    if (knodesD[final_last_knode_idx].keys[thid] == final_end) {
        s_reclen = knodesD[final_last_knode_idx].indices[thid] - s_recstart + 1;
    }
    __syncthreads();

    // Write back to global memory once at the end
    if (thid == 0) {
        currKnodeD[bid] = s_currKnode;
        offsetD[bid] = s_offset;
        lastKnodeD[bid] = s_lastKnode;
        offset_2D[bid] = s_offset_2;
        RecstartD[bid] = s_recstart;
        ReclenD[bid] = s_reclen;
    }
}
