#include <cuda.h>

__global__ void findK(long height, knode * __restrict__ knodesD, long knodes_elem,
                      record * __restrict__ recordsD,
                      long * __restrict__ currKnodeD, long * __restrict__ offsetD,
                      int * __restrict__ keysD,
                      record * __restrict__ ansD) {

    // private thread IDs
    int thid = threadIdx.x;
    int bid  = blockIdx.x;

    // cache frequently used per-block values in registers
    int  key   = keysD[bid];
    long curr  = currKnodeD[bid];
    long off   = offsetD[bid];

    // processtree levels
    for (long lvl = 0; lvl < height; ++lvl) {

        // read current node once per level via register
        knode node_curr = knodesD[curr];
        knode node_off  = knodesD[off];

        int k_left  = node_curr.keys[thid];
        int k_right = node_curr.keys[thid + 1];

        // if value is between the two keys
        if (k_left <= key && k_right > key) {
            long child_index = node_off.indices[thid];

            // this conditional statement is inserted to avoid crush due to bug
            // in original code
            if (child_index < knodes_elem) {
                off = child_index;
            }
        }
        __syncthreads();

        // set for next tree level
        if (thid == 0) {
            curr = off;
        }
        __syncthreads();
    }

    // write back updated state for this block
    if (thid == 0) {
        currKnodeD[bid] = curr;
        offsetD[bid]    = off;
    }
    __syncthreads();

    // At this point, we have a candidate leaf node which may contain
    // the target record.  Check each key to hopefully find the record
    knode leaf = knodesD[curr];

    if (leaf.keys[thid] == key) {
        long rec_idx = leaf.indices[thid];
        ansD[bid].value = recordsD[rec_idx].value;
    }
}
