__global__ void findRangeK(long height,

                           knode *knodesD, long knodes_elem,

                           long *currKnodeD, long *offsetD, long *lastKnodeD,
                           long *offset_2D, int *startD, int *endD,
                           int *RecstartD, int *ReclenD) {

    // private thread IDs
    int thid = threadIdx.x;
    int bid = blockIdx.x;

    // Shared memory for current and last node to reduce global memory access
    __shared__ long s_currKnode;
    __shared__ long s_lastKnode;
    __shared__ long s_offset;
    __shared__ long s_offset_2;
    
    // Load current nodes into shared memory once per block
    if (thid == 0) {
        s_currKnode = currKnodeD[bid];
        s_lastKnode = lastKnodeD[bid];
        s_offset = offsetD[bid];
        s_offset_2 = offset_2D[bid];
    }
    __syncthreads();

    // Cache the search keys in registers
    int search_start = startD[bid];
    int search_end = endD[bid];

    // ???
    int i;
    for (i = 0; i < height; i++) {

        // Coalesced memory access: threads read consecutive keys
        long curr_node = s_currKnode;
        long last_node = s_lastKnode;
        
        int curr_key_lower = knodesD[curr_node].keys[thid];
        int curr_key_upper = knodesD[curr_node].keys[thid + 1];
        
        if ((curr_key_lower <= search_start) && (curr_key_upper > search_start)) {
            // this conditional statement is inserted to avoid crush due to but
            // in original code
            // "offset[bid]" calculated below that later addresses part of
            // knodes goes outside of its bounds cause segmentation fault
            // more specifically, values saved into knodes->indices in the main
            // function are out of bounds of knodes that they address
            long next_index = knodesD[curr_node].indices[thid];
            if (next_index < knodes_elem) {
                s_offset = next_index;
            }
        }
        
        int last_key_lower = knodesD[last_node].keys[thid];
        int last_key_upper = knodesD[last_node].keys[thid + 1];
        
        if ((last_key_lower <= search_end) && (last_key_upper > search_end)) {
            // this conditional statement is inserted to avoid crush due to but
            // in original code
            // "offset_2[bid]" calculated below that later addresses part of
            // knodes goes outside of its bounds cause segmentation fault
            // more specifically, values saved into knodes->indices in the main
            // function are out of bounds of knodes that they address
            long next_index_2 = knodesD[last_node].indices[thid];
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

    // Find the index of the starting record
    long final_curr_node = s_currKnode;
    if (knodesD[final_curr_node].keys[thid] == search_start) {
        RecstartD[bid] = knodesD[final_curr_node].indices[thid];
    }
    __syncthreads();

    // Find the index of the ending record
    long final_last_node = s_lastKnode;
    if (knodesD[final_last_node].keys[thid] == search_end) {
        ReclenD[bid] =
            knodesD[final_last_node].indices[thid] - RecstartD[bid] + 1;
    }
}
