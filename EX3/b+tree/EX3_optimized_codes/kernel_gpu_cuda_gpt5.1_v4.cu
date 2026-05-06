__global__ void findK(long height, knode *knodesD, long knodes_elem,
                      record *recordsD,

                      long *currKnodeD, long *offsetD, int *keysD,
                      record *ansD) {

    // private thread IDs
    const int thid = threadIdx.x;
    const int bid  = blockIdx.x;

    // cache frequently used values in registers
    int   key    = keysD[bid];
    long  curr   = currKnodeD[bid];
    long  offset = offsetD[bid];

    // loop over tree levels
    for (long level = 0; level < height; ++level) {

        // cache node indices and keys in registers
        const long  currIdx   = curr;
        const long  offsetIdx = offset;

        const int   keyLeft   = knodesD[currIdx].keys[thid];
        const int   keyRight  = knodesD[currIdx].keys[thid + 1];
        const long  childIdx  = knodesD[offsetIdx].indices[thid];

        // if value is between the two keys
        if (keyLeft <= key && keyRight > key) {
            // this conditional statement is inserted to avoid crush due to bug
            // in original code
            // "offset[bid]" calculated below that addresses knodes[] in the
            // next iteration goes outside of its bounds cause segmentation
            // fault
            // more specifically, values saved into knodes->indices in the main
            // function are out of bounds of knodes that they address
            if (childIdx < knodes_elem) {
                offset = childIdx;
            }
        }
        __syncthreads();

        // set for next tree level
        if (thid == 0) {
            curr = offset;
        }
        __syncthreads();
    }

    // write back updated traversal state
    if (thid == 0) {
        currKnodeD[bid] = curr;
        offsetD[bid]    = offset;
    }
    __syncthreads();

    // At this point, we have a candidate leaf node which may contain
    // the target record.  Check each key to hopefully find the record
    const long leafIdx = curr;
    if (knodesD[leafIdx].keys[thid] == key) {
        const long recIdx = knodesD[leafIdx].indices[thid];
        ansD[bid].value   = recordsD[recIdx].value;
    }
}
