#include <cuda.h>
#include <cuda_runtime.h>

__global__ void findK(long height, knode *knodesD, long knodes_elem,
                      record *recordsD,

                      long *currKnodeD, long *offsetD, int *keysD,
                      record *ansD) {

    // private thread IDs
    int thid = threadIdx.x;
    int bid = blockIdx.x;

    // cache frequently used values
    const int key_q = __ldg(&keysD[bid]);

    // shared memory for current and next knode index per block
    __shared__ long s_currKnode;
    __shared__ long s_offset;

    // initialize shared current knode from global memory
    if (thid == 0) {
        s_currKnode = currKnodeD[bid];
        s_offset    = offsetD[bid];
    }
    __syncthreads();

    // process tree levels
    for (long i = 0; i < height; i++) {

        // load current node index from shared
        long curr_idx = s_currKnode;

        // load keys for this node; use read-only cache
        int key_left  = __ldg(&knodesD[curr_idx].keys[thid]);
        int key_right = __ldg(&knodesD[curr_idx].keys[thid + 1]);

        // if value is between the two keys
        if (key_left <= key_q && key_right > key_q) {

            // load candidate child index through read-only cache
            long cand_offset = __ldg(&s_offset);
            int child_index  = __ldg(&knodesD[cand_offset].indices[thid]);

            // this conditional statement is inserted to avoid crush due to bug
            // in original code
            // "offset[bid]" calculated below that addresses knodes[] in the
            // next iteration goes outside of its bounds cause segmentation
            // fault
            // more specifically, values saved into knodes->indices in the main
            // function are out of bounds of knodes that they address
            if (child_index < knodes_elem) {
                cand_offset = child_index;
            }

            // update shared offset
            s_offset = cand_offset;
        }
        __syncthreads();

        // set for next tree level
        if (thid == 0) {
            s_currKnode      = s_offset;
            currKnodeD[bid]  = s_currKnode;
            offsetD[bid]     = s_offset;
        }
        __syncthreads();
    }

    // final node index from shared
    long leaf_idx = s_currKnode;

    // At this point, we have a candidate leaf node which may contain
    // the target record.  Check each key to hopefully find the record
    int leaf_key = __ldg(&knodesD[leaf_idx].keys[thid]);
    if (leaf_key == key_q) {
        int rec_index = __ldg(&knodesD[leaf_idx].indices[thid]);
        ansD[bid].value = recordsD[rec_index].value;
    }
}
