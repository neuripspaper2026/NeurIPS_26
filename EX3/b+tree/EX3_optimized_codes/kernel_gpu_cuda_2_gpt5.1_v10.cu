#include <cuda.h>
#include <cuda_runtime.h>

struct knode {
    int  keys[32];      // must match original definition
    long indices[32];   // must match original definition
};

struct record {
    int value;          // must match original definition
};

__global__ void findRangeK(long height,

                           knode *knodesD, long knodes_elem,

                           long *currKnodeD, long *offsetD, long *lastKnodeD,
                           long *offset_2D, int *startD, int *endD,
                           int *RecstartD, int *ReclenD) {

    int thid = threadIdx.x;
    int bid  = blockIdx.x;

    extern __shared__ int sMem[];
    // Layout: [ keys_curr (blockDim.x+1) | keys_last (blockDim.x+1) ]
    // Reinterpret remaining shared memory for indices
    int  *s_keys_curr = sMem;
    int  *s_keys_last = &s_keys_curr[blockDim.x + 1];

    long *s_indices_curr = (long*)&s_keys_last[blockDim.x + 1];
    long *s_indices_last = &s_indices_curr[blockDim.x];

    // Cache per-block query range in registers
    int s_val = startD[bid];
    int e_val = endD[bid];

    // process tree levels
    for (long level = 0; level < height; level++) {

        // Broadcast current node indices via warp shuffle (minimizes global loads)
        int currNode = (thid == 0) ? (int)currKnodeD[bid] : 0;
        int lastNode = (thid == 0) ? (int)lastKnodeD[bid] : 0;
        __syncthreads();
        currNode = __shfl_sync(0xffffffff, currNode, 0);
        lastNode = __shfl_sync(0xffffffff, lastNode, 0);

        // Cooperative load of keys/indices for current and last nodes into shared memory
        if (thid < blockDim.x) {
            s_keys_curr[thid]    = knodesD[currNode].keys[thid];
            s_indices_curr[thid] = knodesD[currNode].indices[thid];

            s_keys_last[thid]    = knodesD[lastNode].keys[thid];
            s_indices_last[thid] = knodesD[lastNode].indices[thid];
        }

        // Load guard elements for comparison at thid + 1
        if (thid == blockDim.x - 1) {
            s_keys_curr[thid + 1] = knodesD[currNode].keys[thid + 1];
            s_keys_last[thid + 1] = knodesD[lastNode].keys[thid + 1];
        }
        __syncthreads();

        // Search in current node for start
        if (s_keys_curr[thid] <= s_val && s_keys_curr[thid + 1] > s_val) {
            long nextIndex = s_indices_curr[thid];
            if (nextIndex < knodes_elem) {
                offsetD[bid] = nextIndex;
            }
        }

        // Search in last node for end
        if (s_keys_last[thid] <= e_val && s_keys_last[thid + 1] > e_val) {
            long nextIndex2 = s_indices_last[thid];
            if (nextIndex2 < knodes_elem) {
                offset_2D[bid] = nextIndex2;
            }
        }
        __syncthreads();

        // set for next tree level
        if (thid == 0) {
            currKnodeD[bid] = offsetD[bid];
            lastKnodeD[bid] = offset_2D[bid];
        }
        __syncthreads();
    }

    // Broadcast final node indices
    int finalCurr = (thid == 0) ? (int)currKnodeD[bid] : 0;
    int finalLast = (thid == 0) ? (int)lastKnodeD[bid] : 0;
    __syncthreads();
    finalCurr = __shfl_sync(0xffffffff, finalCurr, 0);
    finalLast = __shfl_sync(0xffffffff, finalLast, 0);

    // Reuse shared memory for final node keys/indices
    if (thid < blockDim.x) {
        s_keys_curr[thid]    = knodesD[finalCurr].keys[thid];
        s_indices_curr[thid] = knodesD[finalCurr].indices[thid];

        s_keys_last[thid]    = knodesD[finalLast].keys[thid];
        s_indices_last[thid] = knodesD[finalLast].indices[thid];
    }
    __syncthreads();

    // Find the index of the starting record
    if (s_keys_curr[thid] == s_val) {
        RecstartD[bid] = (int)s_indices_curr[thid];
    }
    __syncthreads();

    // Find the index of the ending record and compute length
    if (s_keys_last[thid] == e_val) {
        ReclenD[bid] = (int)(s_indices_last[thid] - RecstartD[bid] + 1);
    }
}
