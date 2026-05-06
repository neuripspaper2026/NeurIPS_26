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

    // Cache frequently accessed values in registers
    int start_key = startD[bid];
    int end_key = endD[bid];

    // Process tree levels
    for (int i = 0; i < height; i++) {

        long curr_knode = shared_currKnode[0];
        long last_knode = shared_lastKnode[0];

        if ((knodesD[curr_knode].keys[thid] <= start_key) &&
            (knodesD[curr_knode].keys[thid + 1] > start_key)) {
            // this conditional statement is inserted to avoid crush due to but
            // in original code
            // "offset[bid]" calculated below that later addresses part of
            // knodes goes outside of its bounds cause segmentation fault
            // more specifically, values saved into knodes->indices in the main
            // function are out of bounds of knodes that they address
            if (knodesD[curr_knode].indices[thid] < knodes_elem) {
                shared_offset[0] = knodesD[curr_knode].indices[thid];
            }
        }
        if ((knodesD[last_knode].keys[thid] <= end_key) &&
            (knodesD[last_knode].keys[thid + 1] > end_key)) {
            // this conditional statement is inserted to avoid crush due to but
            // in original code
            // "offset_2[bid]" calculated below that later addresses part of
            // knodes goes outside of its bounds cause segmentation fault
            // more specifically, values saved into knodes->indices in the main
            // function are out of bounds of knodes that they address
            if (knodesD[last_knode].indices[thid] < knodes_elem) {
                shared_offset_2[0] = knodesD[last_knode].indices[thid];
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
    long final_curr_knode = shared_currKnode[0];
    if (knodesD[final_curr_knode].keys[thid] == start_key) {
        RecstartD[bid] = knodesD[final_curr_knode].indices[thid];
    }
    __syncthreads();

    // Find the index of the ending record
    long final_last_knode = shared_lastKnode[0];
    if (knodesD[final_last_knode].keys[thid] == end_key) {
        ReclenD[bid] =
            knodesD[final_last_knode].indices[thid] - RecstartD[bid] + 1;
    }
}
