#include <cuda.h>
#include <cuda_runtime.h>

__global__ void findRangeK(long height,

                           knode *knodesD, long knodes_elem,

                           long *currKnodeD, long *offsetD, long *lastKnodeD,
                           long *offset_2D, int *startD, int *endD,
                           int *RecstartD, int *ReclenD) {

    // private thread IDs
    const int thid = threadIdx.x;
    const int bid  = blockIdx.x;

    // cache query range for this block in registers (read-only)
    const int start_q = __ldg(&startD[bid]);
    const int end_q   = __ldg(&endD[bid]);

    // shared memory for current/last knode indices and offsets
    __shared__ long s_currKnode;
    __shared__ long s_lastKnode;
    __shared__ long s_offset;
    __shared__ long s_offset2;

    // initialize shared state from global memory
    if (thid == 0) {
        s_currKnode = currKnodeD[bid];
        s_lastKnode = lastKnodeD[bid];
        s_offset    = offsetD[bid];
        s_offset2   = offset_2D[bid];
    }
    __syncthreads();

    // tree traversal over height levels
    for (long i = 0; i < height; i++) {

        // load current node indices from shared
        long curr_idx = s_currKnode;
        long last_idx = s_lastKnode;

        // load keys for current and last node; use read-only cache
        int key_curr_left  = __ldg(&knodesD[curr_idx].keys[thid]);
        int key_curr_right = __ldg(&knodesD[curr_idx].keys[thid + 1]);

        int key_last_left  = __ldg(&knodesD[last_idx].keys[thid]);
        int key_last_right = __ldg(&knodesD[last_idx].keys[thid + 1]);

        // search for start key range in current node
        if (key_curr_left <= start_q && key_curr_right > start_q) {

            int child_index =
                __ldg(&knodesD[curr_idx].indices[thid]);

            // guard against out-of-bounds indices, as in original code
            if (child_index < knodes_elem) {
                s_offset = child_index;
            }
        }

        // search for end key range in last node
        if (key_last_left <= end_q && key_last_right > end_q) {

            int child_index2 =
                __ldg(&knodesD[last_idx].indices[thid]);

            // guard against out-of-bounds indices, as in original code
            if (child_index2 < knodes_elem) {
                s_offset2 = child_index2;
            }
        }
        __syncthreads();

        // set for next tree level and update global arrays
        if (thid == 0) {
            s_currKnode    = s_offset;
            s_lastKnode    = s_offset2;
            currKnodeD[bid] = s_currKnode;
            lastKnodeD[bid] = s_lastKnode;
            offsetD[bid]    = s_offset;
            offset_2D[bid]  = s_offset2;
        }
        __syncthreads();
    }

    // final node indices from shared
    long leaf_start_idx = s_currKnode;
    long leaf_end_idx   = s_lastKnode;

    // Find the index of the starting record
    int leaf_start_key = __ldg(&knodesD[leaf_start_idx].keys[thid]);
    if (leaf_start_key == start_q) {
        int rec_index = __ldg(&knodesD[leaf_start_idx].indices[thid]);
        RecstartD[bid] = rec_index;
    }
    __syncthreads();

    // Find the index of the ending record
    int leaf_end_key = __ldg(&knodesD[leaf_end_idx].keys[thid]);
    if (leaf_end_key == end_q) {
        int end_rec_index = __ldg(&knodesD[leaf_end_idx].indices[thid]);
        ReclenD[bid] = end_rec_index - RecstartD[bid] + 1;
    }
}
