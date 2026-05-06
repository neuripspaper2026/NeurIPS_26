#ifndef _KERNEL2_H_
#define _KERNEL2_H_

__global__ void Kernel2(bool *g_graph_mask, bool *g_updating_graph_mask,
                        bool *g_graph_visited, bool *g_over, int no_of_nodes) {
    int tid = blockIdx.x * MAX_THREADS_PER_BLOCK + threadIdx.x;
    
    if (tid < no_of_nodes) {
        // Load updating mask with coalesced access
        bool updating = g_updating_graph_mask[tid];
        
        if (updating) {
            // Perform all coalesced writes
            g_graph_mask[tid] = true;
            g_graph_visited[tid] = true;
            g_updating_graph_mask[tid] = false;
            
            // Use atomic operation for global flag
            atomicExch((int*)g_over, 1);
        }
    }
}

#endif
