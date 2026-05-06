#ifndef _KERNEL2_H_
#define _KERNEL2_H_

__global__ void Kernel2(bool *g_graph_mask, bool *g_updating_graph_mask,
                        bool *g_graph_visited, bool *g_over, int no_of_nodes) {
    int tid = blockIdx.x * MAX_THREADS_PER_BLOCK + threadIdx.x;
    
    if (tid < no_of_nodes) {
        // Load updating mask value once into register
        bool needs_update = g_updating_graph_mask[tid];
        
        if (needs_update) {
            // Perform all writes with coalesced memory access
            g_graph_mask[tid] = true;
            g_graph_visited[tid] = true;
            g_updating_graph_mask[tid] = false;
            
            // Use warp-level primitive to detect if any thread needs to set g_over
            // This reduces atomic contention on g_over
            unsigned mask = __ballot_sync(0xffffffff, needs_update);
            if (mask != 0 && (threadIdx.x & 31) == 0) {
                *g_over = true;
            }
        }
    }
}

#endif
