__global__ void findK(long height, knode *knodesD, long knodes_elem,
                      record *recordsD,

                      long *currKnodeD, long *offsetD, int *keysD,
                      record *ansD) {

    // private thread IDs
    int thid = threadIdx.x;
    int bid = blockIdx.x;

    // shared memory for caching current knode and offset
    __shared__ long shared_currKnode;
    __shared__ long shared_offset;

    // Initialize shared memory
    if (thid == 0) {
        shared_currKnode = currKnodeD[bid];
        shared_offset = offsetD[bid];
    }
    __syncthreads();

    // processtree levels
    for (int i = 0; i < height; i++) {

        // if value is between the two keys
        if ((knodesD[shared_currKnode].keys[thid]) <= keysD[bid] &&
            (knodesD[shared_currKnode].keys[thid + 1] > keysD[bid])) {
            // this conditional statement is inserted to avoid crush due to but
            // in original code
            // "offset[bid]" calculated below that addresses knodes[] in the
            // next iteration goes outside of its bounds cause segmentation
            // fault
            // more specifically, values saved into knodes->indices in the main
            // function are out of bounds of knodes that they address
            if (knodesD[shared_offset].indices[thid] < knodes_elem) {
                shared_offset = knodesD[shared_offset].indices[thid];
            }
        }
        __syncthreads();

        // set for next tree level
        if (thid == 0) {
            shared_currKnode = shared_offset;
        }
        __syncthreads();
    }

    // At this point, we have a candidate leaf node which may contain
    // the target record.  Check each key to hopefully find the record
    if (knodesD[shared_currKnode].keys[thid] == keysD[bid]) {
        ansD[bid].value =
            recordsD[knodesD[shared_currKnode].indices[thid]].value;
    }
}
