#include <cuda.h>

__global__ void findRangeK(long height,

                           knode * __restrict__ knodesD, long knodes_elem,

                           long * __restrict__ currKnodeD, long * __restrict__ offsetD, long * __restrict__ lastKnodeD,
                           long * __restrict__ offset_2D, int * __restrict__ startD, int * __restrict__ endD,
                           int * __restrict__ RecstartD, int * __restrict__ ReclenD) {

    // private thread IDs
    int thid = threadIdx.x;
    int bid  = blockIdx.x;

    // cache frequently used per-block values in registers
    int  start_key = startD[bid];
    int  end_key   = endD[bid];

    long curr  = currKnodeD[bid];
    long last  = lastKnodeD[bid];
    long off   = offsetD[bid];
    long off2  = offset_2D[bid];

    // tree traversal over height levels
    for (long lvl = 0; lvl < height; ++lvl) {

        // load current nodes to registers once per level
        knode node_curr = knodesD[curr];
        knode node_last = knodesD[last];

        int k_curr_left   = node_curr.keys[thid];
        int k_curr_right  = node_curr.keys[thid + 1];
        int k_last_left   = node_last.keys[thid];
        int k_last_right  = node_last.keys[thid + 1];

        // search for start key range
        if (k_curr_left <= start_key && k_curr_right > start_key) {
            long child_index = node_curr.indices[thid];
            // avoid out-of-bounds
            if (child_index < knodes_elem) {
                off = child_index;
            }
        }

        // search for end key range
        if (k_last_left <= end_key && k_last_right > end_key) {
            long child_index2 = node_last.indices[thid];
            // avoid out-of-bounds
            if (child_index2 < knodes_elem) {
                off2 = child_index2;
            }
        }

        __syncthreads();

        // set for next tree level (single thread updates, others observe)
        if (thid == 0) {
            curr = off;
            last = off2;
        }
        __syncthreads();
    }

    // write back updated traversal state
    if (thid == 0) {
        currKnodeD[bid]  = curr;
        lastKnodeD[bid]  = last;
        offsetD[bid]     = off;
        offset_2D[bid]   = off2;
    }
    __syncthreads();

    // load final nodes for range boundaries
    knode node_curr_final = knodesD[curr];
    knode node_last_final = knodesD[last];

    int key_curr = node_curr_final.keys[thid];
    int key_last = node_last_final.keys[thid];

    // Find the index of the starting record
    if (key_curr == start_key) {
        RecstartD[bid] = node_curr_final.indices[thid];
    }
    __syncthreads();

    // Find the index of the ending record
    if (key_last == end_key) {
        ReclenD[bid] = node_last_final.indices[thid] - RecstartD[bid] + 1;
    }
}
