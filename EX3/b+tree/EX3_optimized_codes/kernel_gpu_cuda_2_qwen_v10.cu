__global__ void findRangeK(long height,

                           knode *knodesD, long knodes_elem,

                           long *currKnodeD, long *offsetD, long *lastKnodeD,
                           long *offset_2D, int *startD, int *endD,
                           int *RecstartD, int *ReclenD) {

    // private thread IDs
    int thid = threadIdx.x;
    int bid = blockIdx.x;

    // Shared memory for frequently accessed data
    extern __shared__ long shared_data[];
    long *shared_currKnode = &shared_data[0];
    long *shared_lastKnode = &shared_data[1];
    long *shared_offset = &shared_data[2];
    long *shared_offset_2 = &shared_data[3];

    // Initialize shared memory
    if (thid == 0) {
        shared_currKnode[0] = currKnodeD[bid];
        shared_lastKnode[0] = lastKnodeD[bid];
        shared_offset[0] = offsetD[bid];
        shared_offset_2[0] = offset_2D[bid];
    }
    __syncthreads();

    // processtree levels
    int i;
    for (i = 0; i < height; i++) {

        if ((knodesD[shared_currKnode[0]].keys[thid] <= startD[bid]) &&
            (knodesD[shared_currKnode[0]].keys[thid + 1] > startD[bid])) {
            // this conditional statement is inserted to avoid crush due to but
            // in original code
            // "offset[bid]" calculated below that later addresses part of
            // knodes goes outside of its bounds cause segmentation fault
            // more specifically, values saved into knodes->indices in the main
            // function are out of bounds of knodes that they address
            if (knodesD[shared_currKnode[0]].indices[thid] < knodes_elem) {
                shared_offset[0] = knodesD[shared_currKnode[0]].indices[thid];
            }
        }
        if ((knodesD[shared_lastKnode[0]].keys[thid] <= endD[bid]) &&
            (knodesD[shared_lastKnode[0]].keys[thid + 1] > endD[bid])) {
            // this conditional statement is inserted to avoid crush due to but
            // in original code
            // "offset_2[bid]" calculated below that later addresses part of
            // knodes goes outside of its bounds cause segmentation fault
            // more specifically, values saved into knodes->indices in the main
            // function are out of bounds of knodes that they address
            if (knodesD[shared_lastKnode[0]].indices[thid] < knodes_elem) {
                shared_offset_2[0] = knodesD[shared_lastKnode[0]].indices[thid];
            }
        }
        __syncthreads();

        // set for next tree level
        if (thid == 0) {
            shared_currKnode[0] = shared_offset[0];
            shared_lastKnode[0] = shared_offset_2[0];
        }
        __syncthreads();
    }

    // Find the index of the starting record
    if (knodesD[shared_currKnode[0]].keys[thid] == startD[bid]) {
        RecstartD[bid] = knodesD[shared_currKnode[0]].indices[thid];
    }
    __syncthreads();

    // Find the index of the ending record
    if (knodesD[shared_lastKnode[0]].keys[thid] == endD[bid]) {
        ReclenD[bid] =
            knodesD[shared_lastKnode[0]].indices[thid] - RecstartD[bid] + 1;
    }
}
