#include <cuda.h>
#include <cuda_runtime.h>

struct knode {
    int keys[32];     // assuming max keys per node (example; actual definition should match original)
    long indices[32]; // assuming corresponding indices (example; actual definition should match original)
};

struct record {
    int value;        // example; actual definition should match original
};

__global__ void findK(long height, knode *knodesD, long knodes_elem,
                      record *recordsD,
                      long *currKnodeD, long *offsetD, int *keysD,
                      record *ansD) {

    int thid = threadIdx.x;
    int bid  = blockIdx.x;

    extern __shared__ int sMem[];
    int  *s_keys    = sMem;
    long *s_indices = (long*)&s_keys[blockDim.x + 1];
    int  *s_query   = (int*)&s_indices[blockDim.x];

    // Cache query key per block (all threads in block work on same bid)
    if (thid == 0) {
        s_query[0] = keysD[bid];
    }
    __syncthreads();
    int queryKey = s_query[0];

    // process tree levels
    for (long level = 0; level < height; level++) {

        // Load current node index for this block (broadcast via shared memory)
        int currNode = (thid == 0) ? (int)currKnodeD[bid] : 0;
        __syncthreads();
        currNode = __shfl_sync(0xffffffff, currNode, 0);

        // Cooperative load of keys and indices for current node into shared memory
        if (thid < blockDim.x) {
            s_keys[thid]    = knodesD[currNode].keys[thid];
            s_indices[thid] = knodesD[currNode].indices[thid];
        }
        if (thid == blockDim.x - 1) {
            // Guard element for comparison at thid + 1
            s_keys[thid + 1] = knodesD[currNode].keys[thid + 1];
        }
        __syncthreads();

        // if value is between the two keys
        if (s_keys[thid] <= queryKey && s_keys[thid + 1] > queryKey) {
            long nextIndex = s_indices[thid];
            if (nextIndex < knodes_elem) {
                offsetD[bid] = nextIndex;
            }
        }
        __syncthreads();

        // set for next tree level
        if (thid == 0) {
            currKnodeD[bid] = offsetD[bid];
        }
        __syncthreads();
    }

    // At this point, we have a candidate leaf node which may contain
    // the target record. Check each key to hopefully find the record
    int finalNode = (thid == 0) ? (int)currKnodeD[bid] : 0;
    __syncthreads();
    finalNode = __shfl_sync(0xffffffff, finalNode, 0);

    // Reuse shared memory for final node keys and indices to reduce global traffic
    if (thid < blockDim.x) {
        s_keys[thid]    = knodesD[finalNode].keys[thid];
        s_indices[thid] = knodesD[finalNode].indices[thid];
    }
    __syncthreads();

    if (s_keys[thid] == queryKey) {
        ansD[bid].value = recordsD[s_indices[thid]].value;
    }
}
