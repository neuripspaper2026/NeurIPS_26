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
        
        // Cache current knode pointer to reduce global memory accesses
        long curr_knode = s_currKnode;
        
        // Load keys into registers for comparison
        int key_lower = knodesD[curr_knode].keys[thid];
        int key_upper = knodesD[curr_knode].keys[thid + 1];
        
        // if value is between the two keys
        if (key_lower <= s_key && key_upper > s_key) {
            // this conditional statement is inserted to avoid crush due to but
            // in original code
            // "offset[bid]" calculated below that addresses knodes[] in the
            // next iteration goes outside of its bounds cause segmentation
            // fault
            // more specifically, values saved into knodes->indices in the main
            // function are out of bounds of knodes that they address
            long next_index = knodesD[s_offset].indices[thid];
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
    long final_knode = s_currKnode;
    int final_key = knodesD[final_knode].keys[thid];
    
    if (final_key == s_key) {
        long record_index = knodesD[final_knode].indices[thid];
        ansD[bid].value = recordsD[record_index].value;
    }
}
