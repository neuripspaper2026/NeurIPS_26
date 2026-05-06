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

        // Load knode keys into registers for this iteration
        int key_curr_start = knodesD[s_currKnode].keys[thid];
        int key_next_start = knodesD[s_currKnode].keys[thid + 1];
        
        if ((key_curr_start <= s_start) && (key_next_start > s_start)) {
            // Load the index once
            long candidate_offset = knodesD[s_currKnode].indices[thid];
            
            // this conditional statement is inserted to avoid crush due to but
            // in original code
            // "offset[bid]" calculated below that later addresses part of
            // knodes goes outside of its bounds cause segmentation fault
            // more specifically, values saved into knodes->indices in the main
            // function are out of bounds of knodes that they address
            if (candidate_offset < knodes_elem) {
                s_offset = candidate_offset;
            }
        }
        
        // Load knode keys for end search
        int key_curr_end = knodesD[s_lastKnode].keys[thid];
        int key_next_end = knodesD[s_lastKnode].keys[thid + 1];
        
        if ((key_curr_end <= s_end) && (key_next_end > s_end)) {
            // Load the index once
            long candidate_offset_2 = knodesD[s_lastKnode].indices[thid];
            
            // this conditional statement is inserted to avoid crush due to but
            // in original code
            // "offset_2[bid]" calculated below that later addresses part of
            // knodes goes outside of its bounds cause segmentation fault
            // more specifically, values saved into knodes->indices in the main
            // function are out of bounds of knodes that they address
            if (candidate_offset_2 < knodes_elem) {
                s_offset_2 = candidate_offset_2;
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

    // Find the index of the starting record
    int final_key_start = knodesD[s_currKnode].keys[thid];
    if (final_key_start == s_start) {
        RecstartD[bid] = knodesD[s_currKnode].indices[thid];
    }
    __syncthreads();

    // Find the index of the ending record
    int final_key_end = knodesD[s_lastKnode].keys[thid];
    if (final_key_end == s_end) {
        ReclenD[bid] = knodesD[s_lastKnode].indices[thid] - RecstartD[bid] + 1;
    }
    
    // Write back shared memory to global memory at the end
    if (thid == 0) {
        currKnodeD[bid] = s_currKnode;
        offsetD[bid] = s_offset;
        lastKnodeD[bid] = s_lastKnode;
        offset_2D[bid] = s_offset_2;
    }
}
