#include <cuda.h>

__global__ void findRangeK(long height,

                           knode * __restrict__ knodesD, long knodes_elem,

                           long * __restrict__ currKnodeD, long * __restrict__ offsetD, long * __restrict__ lastKnodeD,
                           long * __restrict__ offset_2D, int * __restrict__ startD, int * __restrict__ endD,
                           int * __restrict__ RecstartD, int * __restrict__ ReclenD) {

    // private thread IDs
    const int thid = threadIdx.x;
    const int bid  = blockIdx.x;

    // cache frequently used per-block data in registers
    int  start_key = startD[bid];
    int  end_key   = endD[bid];
    long curr      = currKnodeD[bid];
    long last      = lastKnodeD[bid];
    long off1      = offsetD[bid];
    long off2      = offset_2D[bid];

    // process tree levels
    for (long level = 0; level < height; ++level) {

        // load knodes pointers once per level
        knode * __restrict__ kn_curr = &knodesD[curr];
        knode * __restrict__ kn_last = &knodesD[last];

        // read keys into registers to avoid repeated global loads
        int currKeyL = kn_curr->keys[thid];
        int currKeyR = kn_curr->keys[thid + 1];
        int lastKeyL = kn_last->keys[thid];
        int lastKeyR = kn_last->keys[thid + 1];

        // traversal for start_key
        if (currKeyL <= start_key && currKeyR > start_key) {
            long child_idx = kn_curr->indices[thid];
            // bounds check to avoid out-of-range access
            if (child_idx < knodes_elem) {
                off1 = child_idx;
            }
        }

        // traversal for end_key
        if (lastKeyL <= end_key && lastKeyR > end_key) {
            long child_idx2 = kn_last->indices[thid];
            // bounds check to avoid out-of-range access
            if (child_idx2 < knodes_elem) {
                off2 = child_idx2;
            }
        }

        __syncthreads();

        // set for next tree level (single thread updates shared state)
        if (thid == 0) {
            curr = off1;
            last = off2;
        }
        __syncthreads();
    }

    // write back updated node indices and offsets to global memory
    if (thid == 0) {
        currKnodeD[bid] = curr;
        lastKnodeD[bid] = last;
        offsetD[bid]    = off1;
        offset_2D[bid]  = off2;
    }
    __syncthreads();

    // Find the index of the starting record
    knode * __restrict__ kn_curr_final = &knodesD[curr];
    int finalStartKey = kn_curr_final->keys[thid];
    if (finalStartKey == start_key) {
        RecstartD[bid] = kn_curr_final->indices[thid];
    }
    __syncthreads();

    // Find the index of the ending record
    knode * __restrict__ kn_last_final = &knodesD[last];
    int finalEndKey = kn_last_final->keys[thid];
    if (finalEndKey == end_key) {
        ReclenD[bid] =
            kn_last_final->indices[thid] - RecstartD[bid] + 1;
    }
}
