#include <cuda.h>

__global__ void findK(long height, knode *knodesD, long knodes_elem,
                      record *recordsD,

                      long *currKnodeD, long *offsetD, int *keysD,
                      record *ansD) {

    // private thread IDs
    const int thid = threadIdx.x;
    const int bid  = blockIdx.x;

    // Cache per-block values in registers
    int key      = keysD[bid];
    long curr    = currKnodeD[bid];
    long offset  = offsetD[bid];

    // Using shared memory to reduce redundant global loads of current node
    extern __shared__ long sdata[];
    long *s_keys    = sdata;                       // size >= blockDim.x+1
    long *s_indices = s_keys + (blockDim.x + 1);   // size >= blockDim.x

    // process tree levels
    for (long level = 0; level < height; level++) {

        // Load keys and indices of current node into shared memory
        knode *node = &knodesD[curr];

        if (thid < blockDim.x + 1) {
            s_keys[thid] = node->keys[thid];
        }
        if (thid < blockDim.x) {
            s_indices[thid] = node->indices[thid];
        }
        __syncthreads();

        // if value is between the two keys
        if (s_keys[thid] <= key && s_keys[thid + 1] > key) {
            // this conditional statement is inserted to avoid crush due to bug
            // in original code
            // "offset[bid]" calculated below that addresses knodes[] in the
            // next iteration goes outside of its bounds cause segmentation
            // fault
            // more specifically, values saved into knodes->indices in the main
            // function are out of bounds of knodes that they address
            long child = s_indices[thid];
            if (child < knodes_elem) {
                offset = child;
            }
        }
        __syncthreads();

        // set for next tree level
        if (thid == 0) {
            curr = offset;
            currKnodeD[bid] = curr;
            offsetD[bid]    = offset;
        }
        __syncthreads();

        // Update cached pointer for next iteration
        curr = currKnodeD[bid];
        offset = offsetD[bid];
    }

    // At this point, we have a candidate leaf node which may contain
    // the target record.  Check each key to hopefully find the record

    // Load leaf node keys/indices into shared memory to reduce repeated global loads
    knode *leaf = &knodesD[curr];
    if (thid < blockDim.x + 1) {
        s_keys[thid] = leaf->keys[thid];
    }
    if (thid < blockDim.x) {
        s_indices[thid] = leaf->indices[thid];
    }
    __syncthreads();

    if (s_keys[thid] == key) {
        long rec_idx = s_indices[thid];
        ansD[bid].value = recordsD[rec_idx].value;
    }
}
