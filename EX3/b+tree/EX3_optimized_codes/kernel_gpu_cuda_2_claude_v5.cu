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

        // Cache current knode pointers to reduce global memory accesses
        long curr_knode = s_currKnode;
        long last_knode = s_lastKnode;
        
        // Load keys into registers for comparison
        int curr_key_lower = knodesD[curr_knode].keys[thid];
        int curr_key_upper = knodesD[curr_knode].keys[thid + 1];
        int last_key_lower = knodesD[last_knode].keys[thid];
        int last_key_upper = knodesD[last_knode].keys[thid + 1];

        if ((curr_key_lower <= s_start) && (curr_key_upper > s_start)) {
            // this conditional statement is inserted to avoid crush due to but
            // in original code
            // "offset[bid]" calculated below that later addresses part of
            // knodes goes outside of its bounds cause segmentation fault
            // more specifically, values saved into knodes->indices in the main
            // function are out of bounds of knodes that they address
            long next_index = knodesD[curr_knode].indices[thid];
            if (next_index < knodes_elem) {
                s_offset = next_index;
            }
        }
        if ((last_key_lower <= s_end) && (last_key_upper > s_end)) {
            // this conditional statement is inserted to avoid crush due to but
            // in original code
            // "offset_2[bid]" calculated below that later addresses part of
            // knodes goes outside of its bounds cause segmentation fault
            // more specifically, values saved into knodes->indices in the main
            // function are out of bounds of knodes that they address
            long next_index_2 = knodesD[last_knode].indices[thid];
            if (next_index_2 < knodes_elem) {
                s_offset_2 = next_index_2;
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
    long final_curr_knode = s_currKnode;
    long final_last_knode = s_lastKnode;
    
    // Load final keys into registers
    int final_curr_key = knodesD[final_curr_knode].keys[thid];
    int final_last_key = knodesD[final_last_knode].keys[thid];

    // Find the index of the starting record
    if (final_curr_key == s_start) {
        RecstartD[bid] = knodesD[final_curr_knode].indices[thid];
    }
    __syncthreads();

    // Find the index of the ending record
    if (final_last_key == s_end) {
        ReclenD[bid] =
            knodesD[final_last_knode].indices[thid] - RecstartD[bid] + 1;
    }
}
