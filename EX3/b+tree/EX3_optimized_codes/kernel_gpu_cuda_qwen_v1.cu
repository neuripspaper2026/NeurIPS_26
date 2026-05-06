__global__ void findK(long height, knode *knodesD, long knodes_elem,
                      record *recordsD,

                      long *currKnodeD, long *offsetD, int *keysD,
                      record *ansD) {

    // private thread IDs
    int thid = threadIdx.x;
    int bid = blockIdx.x;

    // Shared memory for caching frequently accessed data
    extern __shared__ long shared_data[];
    long *shared_currKnode = &shared_data[0];
    long *shared_offset = &shared_data[blockDim.x];

    // Initialize shared memory
    if (thid == 0) {
        shared_currKnode[0] = currKnodeD[bid];
        shared_offset[0] = offsetD[bid];
    }
    __syncthreads();

    // processtree levels
    for (int i = 0; i < height; i++) {

        // if value is between the two keys
        if ((knodesD[shared_currKnode[0]].keys[thid]) <= keysD[bid] &&
            (knodesD[shared_currKnode[0]].keys[thid + 1] > keysD[bid])) {
            // this conditional statement is inserted to avoid crush due to but
            // in original code
            // "offset[bid]" calculated below that addresses knodes[] in the
            // next iteration goes outside of its bounds cause segmentation
            // fault
            // more specifically, values saved into knodes->indices in the main
            // function are out of bounds of knodes that they address
            if (knodesD[shared_offset[0]].indices[thid] < knodes_elem) {
                shared_offset[0] = knodesD[shared_offset[0]].indices[thid];
            }
        }
        __syncthreads();

        // set for next tree level
        if (thid == 0) {
            shared_currKnode[0] = shared_offset[0];
        }
        __syncthreads();
    }

    // At this point, we have a candidate leaf node which may contain
    // the target record.  Check each key to hopefully find the record
    if (knodesD[shared_currKnode[0]].keys[thid] == keysD[bid]) {
        ansD[bid].value =
            recordsD[knodesD[shared_currKnode[0]].indices[thid]].value;
    }

    // Write back to global memory
    if (thid == 0) {
        currKnodeD[bid] = shared_currKnode[0];
        offsetD[bid] = shared_offset[0];
    }
}
