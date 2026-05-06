#ifndef _KERNEL2_H_
#define _KERNEL2_H_

__global__ void Kernel2(bool *g_graph_mask, bool *g_updating_graph_mask,
                        bool *g_graph_visited, bool *g_over, int no_of_nodes) {
    int tid = blockIdx.x * MAX_THREADS_PER_BLOCK + threadIdx.x;
    
    if (tid < no_of_nodes) {
        // Load updating mask into register to avoid redundant global memory access
        bool updating = g_updating_graph_mask[tid];
        
        if (updating) {
            // Coalesced writes - group all writes together
            g_graph_mask[tid] = true;
            g_graph_visited[tid] = true;
            g_updating_graph_mask[tid] = false;
            
            // Use warp-level primitive to reduce atomic contention
            // Only one thread per warp writes to g_over
            if (__any_sync(0xffffffff, updating)) {
                if ((threadIdx.x & 31) == __ffs(__ballot_sync(0xffffffff, updating)) - 1) {
                    *g_over = true;
                }
            }
        }
    }
}

#endif
