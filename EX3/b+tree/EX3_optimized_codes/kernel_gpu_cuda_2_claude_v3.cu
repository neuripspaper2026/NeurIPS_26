__global__ void findRangeK(long height,

                           knode *knodesD, long knodes_elem,

                           long *currKnodeD, long *offsetD, long *lastKnodeD,
                           long *offset_2D, int *startD, int *endD,
                           int *RecstartD, int *ReclenD) {

    // private thread IDs
    int thid = threadIdx.x;
    int bid = blockIdx.x;

    // Load frequently accessed data into registers
    int start_val = startD[bid];
    int end_val = endD[bid];

    // Shared memory for collaborative loading and reducing synchronization overhead
    __shared__ long shared_offset;
    __shared__ long shared_offset_2;
    __shared__ long shared_curr_knode;
    __shared__ long shared_last_knode;

    // Initialize shared memory
    if (thid == 0) {
        shared_curr_knode = currKnodeD[bid];
        shared_offset = offsetD[bid];
        shared_last_knode = lastKnodeD[bid];
        shared_offset_2 = offset_2D[bid];
    }
    __syncthreads();

    // Process tree levels
    int i;
    for (i = 0; i < height; i++) {
        long curr_knode = shared_curr_knode;
        long last_knode = shared_last_knode;
        
        // Coalesced memory access: load keys into registers
        int curr_key_current = knodesD[curr_knode].keys[thid];
        int curr_key_next = knodesD[curr_knode].keys[thid + 1];
        
        // Check if start value is between the two keys
        if (curr_key_current <= start_val && curr_key_next > start_val) {
            long candidate_offset = knodesD[curr_knode].indices[thid];
            // Bounds check
            if (candidate_offset < knodes_elem) {
                // Use atomic operation to ensure only one thread updates
                atomicExch((unsigned long long*)&shared_offset, (unsigned long long)candidate_offset);
            }
        }
        
        // Coalesced memory access for last knode
        int last_key_current = knodesD[last_knode].keys[thid];
        int last_key_next = knodesD[last_knode].keys[thid + 1];
        
        // Check if end value is between the two keys
        if (last_key_current <= end_val && last_key_next > end_val) {
            long candidate_offset_2 = knodesD[last_knode].indices[thid];
            // Bounds check
            if (candidate_offset_2 < knodes_elem) {
                // Use atomic operation to ensure only one thread updates
                atomicExch((unsigned long long*)&shared_offset_2, (unsigned long long)candidate_offset_2);
            }
        }
        __syncthreads();

        // set for next tree level
        if (thid == 0) {
            shared_curr_knode = shared_offset;
            shared_last_knode = shared_offset_2;
        }
        __syncthreads();
    }

    // Final leaf node check
    long curr_knode = shared_curr_knode;
    long last_knode = shared_last_knode;
    
    // Coalesced memory access for final comparison
    int curr_final_key = knodesD[curr_knode].keys[thid];
    
    // Find the index of the starting record
    if (curr_final_key == start_val) {
        long rec_start_index = knodesD[curr_knode].indices[thid];
        atomicExch((unsigned int*)&RecstartD[bid], (unsigned int)rec_start_index);
    }
    __syncthreads();

    // Coalesced memory access for ending record
    int last_final_key = knodesD[last_knode].keys[thid];
    
    // Find the index of the ending record
    if (last_final_key == end_val) {
        long rec_end_index = knodesD[last_knode].indices[thid];
        int rec_len = rec_end_index - RecstartD[bid] + 1;
        atomicExch((unsigned int*)&ReclenD[bid], (unsigned int)rec_len);
    }
    
    // Write back to global memory once at the end
    if (thid == 0) {
        currKnodeD[bid] = shared_curr_knode;
        offsetD[bid] = shared_offset;
        lastKnodeD[bid] = shared_last_knode;
        offset_2D[bid] = shared_offset_2;
    }
}
