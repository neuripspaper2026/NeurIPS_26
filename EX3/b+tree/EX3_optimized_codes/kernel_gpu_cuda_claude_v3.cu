__global__ void findK(long height, knode *knodesD, long knodes_elem,
                      record *recordsD,

                      long *currKnodeD, long *offsetD, int *keysD,
                      record *ansD) {

    // private thread IDs
    int thid = threadIdx.x;
    int bid = blockIdx.x;

    // Load frequently accessed data into registers
    long curr_knode = currKnodeD[bid];
    long offset = offsetD[bid];
    int key = keysD[bid];

    // Shared memory for collaborative loading and reducing synchronization overhead
    __shared__ long shared_offset;
    __shared__ long shared_curr_knode;

    // Initialize shared memory
    if (thid == 0) {
        shared_curr_knode = curr_knode;
        shared_offset = offset;
    }
    __syncthreads();

    // Process tree levels
    int i;
    for (i = 0; i < height; i++) {
        curr_knode = shared_curr_knode;
        
        // Coalesced memory access: load keys into registers
        int key_current = knodesD[curr_knode].keys[thid];
        int key_next = knodesD[curr_knode].keys[thid + 1];
        
        // Check if value is between the two keys
        if (key_current <= key && key_next > key) {
            long candidate_offset = knodesD[curr_knode].indices[thid];
            // Bounds check
            if (candidate_offset < knodes_elem) {
                // Use atomic operation to ensure only one thread updates
                atomicExch((unsigned long long*)&shared_offset, (unsigned long long)candidate_offset);
            }
        }
        __syncthreads();

        // Set for next tree level
        if (thid == 0) {
            shared_curr_knode = shared_offset;
        }
        __syncthreads();
    }

    // Final leaf node check
    curr_knode = shared_curr_knode;
    
    // Coalesced memory access for final comparison
    int final_key = knodesD[curr_knode].keys[thid];
    
    if (final_key == key) {
        long record_index = knodesD[curr_knode].indices[thid];
        // Coalesced read from recordsD
        ansD[bid].value = recordsD[record_index].value;
    }
    
    // Write back to global memory once at the end
    if (thid == 0) {
        currKnodeD[bid] = shared_curr_knode;
        offsetD[bid] = shared_offset;
    }
}
