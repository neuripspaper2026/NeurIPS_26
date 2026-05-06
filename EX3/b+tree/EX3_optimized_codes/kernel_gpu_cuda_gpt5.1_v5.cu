#include <cuda.h>

__global__ void findK(long height, knode * __restrict__ knodesD, long knodes_elem,
                      record * __restrict__ recordsD,
                      long * __restrict__ currKnodeD, long * __restrict__ offsetD,
                      const int * __restrict__ keysD,
                      record * __restrict__ ansD) {

    // private thread IDs
    const int thid = threadIdx.x;
    const int bid  = blockIdx.x;

    // cache frequently used per-block data in registers
    int key   = keysD[bid];
    long curr = currKnodeD[bid];
    long off  = offsetD[bid];

    // processtree levels
    for (long level = 0; level < height; ++level) {

        // load once per loop iteration to improve locality
        knode * __restrict__ kn_curr = &knodesD[curr];
        knode * __restrict__ kn_off  = &knodesD[off];

        // read keys into registers to avoid repeated global loads
        int keyL = kn_curr->keys[thid];
        int keyR = kn_curr->keys[thid + 1];

        // if value is between the two keys
        if (keyL <= key && keyR > key) {
            long child_idx = kn_off->indices[thid];

            // bounds check to avoid out-of-range access
            if (child_idx < knodes_elem) {
                off = child_idx;
            }
        }

        __syncthreads();

        // set for next tree level
        if (thid == 0) {
            curr = off;
        }
        __syncthreads();
    }

    // write back updated node indices to global memory
    if (thid == 0) {
        currKnodeD[bid] = curr;
        offsetD[bid]    = off;
    }
    __syncthreads();

    // At this point, we have a candidate leaf node which may contain
    // the target record.  Check each key to hopefully find the record
    knode * __restrict__ kn_final = &knodesD[curr];
    int finalKey = kn_final->keys[thid];

    if (finalKey == key) {
        long rec_idx = kn_final->indices[thid];
        ansD[bid].value = recordsD[rec_idx].value;
    }
}
