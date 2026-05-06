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
    
    // Initialize shared memory on first thread
    if (thid == 0) {
        s_currKnode = currKnodeD[bid];
        s_offset = offsetD[bid];
        s_key = keysD[bid];
    }
    __syncthreads();

    // processtree levels
    int i;
    for (i = 0; i < height; i++) {
        
        // Load knode keys into registers for this iteration
        int key_curr = knodesD[s_currKnode].keys[thid];
        int key_next = knodesD[s_currKnode].keys[thid + 1];
        
        // if value is between the two keys
        if (key_curr <= s_key && key_next > s_key) {
            // Load the index once
            long candidate_offset = knodesD[s_offset].indices[thid];
            
            // this conditional statement is inserted to avoid crush due to but
            // in original code
            // "offset[bid]" calculated below that addresses knodes[] in the
            // next iteration goes outside of its bounds cause segmentation
            // fault
            // more specifically, values saved into knodes->indices in the main
            // function are out of bounds of knodes that they address
            if (candidate_offset < knodes_elem) {
                s_offset = candidate_offset;
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
    int final_key = knodesD[s_currKnode].keys[thid];
    if (final_key == s_key) {
        long record_idx = knodesD[s_currKnode].indices[thid];
        ansD[bid].value = recordsD[record_idx].value;
    }
    
    // Write back shared memory to global memory at the end
    if (thid == 0) {
        currKnodeD[bid] = s_currKnode;
        offsetD[bid] = s_offset;
    }
}
