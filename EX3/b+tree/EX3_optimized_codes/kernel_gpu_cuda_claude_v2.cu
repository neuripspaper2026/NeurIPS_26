__global__ void findK(long height, knode *knodesD, long knodes_elem,
                      record *recordsD,

                      long *currKnodeD, long *offsetD, int *keysD,
                      record *ansD) {

    // private thread IDs
    int thid = threadIdx.x;
    int bid = blockIdx.x;

    // Shared memory for current node and offset to reduce global memory access
    __shared__ long s_currKnode;
    __shared__ long s_offset;
    
    // Load current node into shared memory once per block
    if (thid == 0) {
        s_currKnode = currKnodeD[bid];
        s_offset = offsetD[bid];
    }
    __syncthreads();

    // Cache the search key in register
    int search_key = keysD[bid];

    // processtree levels
    int i;
    for (i = 0; i < height; i++) {
        // Load current node data
        long curr_node = s_currKnode;
        
        // Coalesced memory access: threads read consecutive keys
        int key_lower = knodesD[curr_node].keys[thid];
        int key_upper = knodesD[curr_node].keys[thid + 1];
        
        // if value is between the two keys
        if (key_lower <= search_key && key_upper > search_key) {
            // Read the index for next level
            long next_index = knodesD[s_offset].indices[thid];
            
            // this conditional statement is inserted to avoid crush due to but
            // in original code
            // "offset[bid]" calculated below that addresses knodes[] in the
            // next iteration goes outside of its bounds cause segmentation
            // fault
            // more specifically, values saved into knodes->indices in the main
            // function are out of bounds of knodes that they address
            if (next_index < knodes_elem) {
                s_offset = next_index;
            }
        }
        __syncthreads();

        // set for next tree level
        if (thid == 0) {
            s_currKnode = s_offset;
        }
        __syncthreads();
    }

    // At this point, we have a candidate leaf node which may contain
    // the target record.  Check each key to hopefully find the record
    long final_node = s_currKnode;
    if (knodesD[final_node].keys[thid] == search_key) {
        long record_idx = knodesD[final_node].indices[thid];
        ansD[bid].value = recordsD[record_idx].value;
    }
}
