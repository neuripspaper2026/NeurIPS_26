__global__ void findK(long height, knode *knodesD, long knodes_elem,
                      record *recordsD,

                      long *currKnodeD, long *offsetD, int *keysD,
                      record *ansD) {

    // private thread IDs
    int thid = threadIdx.x;
    int bid = blockIdx.x;

    // Load frequently accessed data into shared memory
    __shared__ long s_currKnode;
    __shared__ long s_offset;
    __shared__ int s_key;
    
    // Initialize shared memory once per block
    if (thid == 0) {
        s_currKnode = currKnodeD[bid];
        s_offset = offsetD[bid];
        s_key = keysD[bid];
    }
    __syncthreads();

    // processtree levels
    int i;
    for (i = 0; i < height; i++) {
        // Cache current knode pointer for this iteration
        long curr_knode_idx = s_currKnode;
        
        // Coalesced read of keys - threads read consecutive memory locations
        int key_lower = knodesD[curr_knode_idx].keys[thid];
        int key_upper = knodesD[curr_knode_idx].keys[thid + 1];
        long index_val = knodesD[s_offset].indices[thid];

        // if value is between the two keys
        if (key_lower <= s_key && key_upper > s_key) {
            // this conditional statement is inserted to avoid crush due to but
            // in original code
            // "offset[bid]" calculated below that addresses knodes[] in the
            // next iteration goes outside of its bounds cause segmentation
            // fault
            // more specifically, values saved into knodes->indices in the main
            // function are out of bounds of knodes that they address
            if (index_val < knodes_elem) {
                s_offset = index_val;
            }
        }
        __syncthreads();

        // set for next tree level
        if (thid == 0) {
            s_currKnode = s_offset;
        }
        __syncthreads();
    }

    // Write back to global memory once at the end
    if (thid == 0) {
        currKnodeD[bid] = s_currKnode;
        offsetD[bid] = s_offset;
    }

    // At this point, we have a candidate leaf node which may contain
    // the target record.  Check each key to hopefully find the record
    int final_key = knodesD[s_currKnode].keys[thid];
    if (final_key == s_key) {
        long record_idx = knodesD[s_currKnode].indices[thid];
        ansD[bid].value = recordsD[record_idx].value;
    }
}
