__global__ void findK(long height, knode *knodesD, long knodes_elem,
                      record *recordsD,

                      long *currKnodeD, long *offsetD, int *keysD,
                      record *ansD) {

    // private thread IDs
    int thid = threadIdx.x;
    int bid = blockIdx.x;

    // Load frequently accessed data into registers
    long currKnode = currKnodeD[bid];
    long offset = offsetD[bid];
    int key = keysD[bid];

    // processtree levels
    int i;
    for (i = 0; i < height; i++) {

        // Load keys into shared memory for coalesced access
        __shared__ int shared_keys[33];  // ORDER + 1 keys per node
        __shared__ long shared_indices[32];  // ORDER indices per node
        
        if (thid < 33) {
            shared_keys[thid] = knodesD[currKnode].keys[thid];
        }
        if (thid < 32) {
            shared_indices[thid] = knodesD[currKnode].indices[thid];
        }
        __syncthreads();

        // if value is between the two keys
        if ((shared_keys[thid]) <= key &&
            (shared_keys[thid + 1] > key)) {
            // this conditional statement is inserted to avoid crush due to but
            // in original code
            // "offset[bid]" calculated below that addresses knodes[] in the
            // next iteration goes outside of its bounds cause segmentation
            // fault
            // more specifically, values saved into knodes->indices in the main
            // function are out of bounds of knodes that they address
            long next_offset = knodesD[offset].indices[thid];
            if (next_offset < knodes_elem) {
                offset = next_offset;
            }
        }
        __syncthreads();

        // set for next tree level
        if (thid == 0) {
            currKnode = offset;
        }
        __syncthreads();
        
        // Broadcast currKnode to all threads using warp shuffle
        currKnode = __shfl_sync(0xFFFFFFFF, currKnode, 0);
    }

    // At this point, we have a candidate leaf node which may contain
    // the target record.  Check each key to hopefully find the record
    
    // Load final node keys into shared memory
    __shared__ int final_keys[33];
    __shared__ long final_indices[32];
    
    if (thid < 33) {
        final_keys[thid] = knodesD[currKnode].keys[thid];
    }
    if (thid < 32) {
        final_indices[thid] = knodesD[currKnode].indices[thid];
    }
    __syncthreads();
    
    if (final_keys[thid] == key) {
        ansD[bid].value = recordsD[final_indices[thid]].value;
    }
    
    // Write back results to global memory
    if (thid == 0) {
        currKnodeD[bid] = currKnode;
        offsetD[bid] = offset;
    }
}
