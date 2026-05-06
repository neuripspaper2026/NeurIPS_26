#include <cuda.h>

__global__ void findRangeK(long height,

                           knode *knodesD, long knodes_elem,

                           long *currKnodeD, long *offsetD, long *lastKnodeD,
                           long *offset_2D, int *startD, int *endD,
                           int *RecstartD, int *ReclenD) {

    // private thread IDs
    const int thid = threadIdx.x;
    const int bid  = blockIdx.x;

    // Cache per-block scalar values in registers
    int  start    = startD[bid];
    int  end      = endD[bid];
    long curr     = currKnodeD[bid];
    long last     = lastKnodeD[bid];
    long offset   = offsetD[bid];
    long offset_2 = offset_2D[bid];

    // Using shared memory to reduce redundant global loads of current nodes
    extern __shared__ long sdata[];
    long *s_curr_keys    = sdata;                       // size >= blockDim.x+1
    long *s_curr_indices = s_curr_keys + (blockDim.x + 1); // size >= blockDim.x
    long *s_last_keys    = s_curr_indices + blockDim.x; // size >= blockDim.x+1
    long *s_last_indices = s_last_keys + (blockDim.x + 1); // size >= blockDim.x

    // process tree levels
    for (long level = 0; level < height; level++) {

        knode *curr_node = &knodesD[curr];
        knode *last_node = &knodesD[last];

        // Load keys and indices of current node into shared memory
        if (thid < blockDim.x + 1) {
            s_curr_keys[thid] = curr_node->keys[thid];
            s_last_keys[thid] = last_node->keys[thid];
        }
        if (thid < blockDim.x) {
            s_curr_indices[thid] = curr_node->indices[thid];
            s_last_indices[thid] = last_node->indices[thid];
        }
        __syncthreads();

        // for start key
        if (s_curr_keys[thid] <= start && s_curr_keys[thid + 1] > start) {
            long child = s_curr_indices[thid];
            if (child < knodes_elem) {
                offset = child;
            }
        }

        // for end key
        if (s_last_keys[thid] <= end && s_last_keys[thid + 1] > end) {
            long child2 = s_last_indices[thid];
            if (child2 < knodes_elem) {
                offset_2 = child2;
            }
        }
        __syncthreads();

        // set for next tree level
        if (thid == 0) {
            curr           = offset;
            last           = offset_2;
            currKnodeD[bid]   = curr;
            lastKnodeD[bid]   = last;
            offsetD[bid]      = offset;
            offset_2D[bid]    = offset_2;
        }
        __syncthreads();

        // refresh cached values for next iteration
        curr     = currKnodeD[bid];
        last     = lastKnodeD[bid];
        offset   = offsetD[bid];
        offset_2 = offset_2D[bid];
    }

    // After traversal, load leaf nodes to shared memory once more
    knode *curr_leaf = &knodesD[curr];
    knode *last_leaf = &knodesD[last];

    if (thid < blockDim.x + 1) {
        s_curr_keys[thid] = curr_leaf->keys[thid];
        s_last_keys[thid] = last_leaf->keys[thid];
    }
    if (thid < blockDim.x) {
        s_curr_indices[thid] = curr_leaf->indices[thid];
        s_last_indices[thid] = last_leaf->indices[thid];
    }
    __syncthreads();

    // Find the index of the starting record
    if (s_curr_keys[thid] == start) {
        RecstartD[bid] = s_curr_indices[thid];
    }
    __syncthreads();

    // Find the index of the ending record
    if (s_last_keys[thid] == end) {
        ReclenD[bid] = s_last_indices[thid] - RecstartD[bid] + 1;
    }
}
