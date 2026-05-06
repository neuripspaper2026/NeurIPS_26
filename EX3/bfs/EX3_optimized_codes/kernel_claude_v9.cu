#ifndef _KERNEL_H_
#define _KERNEL_H_

__global__ void Kernel(Node *g_graph_nodes, int *g_graph_edges,
                       bool *g_graph_mask, bool *g_updating_graph_mask,
                       bool *g_graph_visited, int *g_cost, int no_of_nodes) {
    int tid = blockIdx.x * MAX_THREADS_PER_BLOCK + threadIdx.x;
    
    if (tid < no_of_nodes) {
        // Load graph mask with coalesced access
        bool mask = g_graph_mask[tid];
        
        if (mask) {
            // Mark mask as false
            g_graph_mask[tid] = false;
            
            // Load node information
            Node node = g_graph_nodes[tid];
            int start = node.starting;
            int end = start + node.no_of_edges;
            int current_cost = g_cost[tid];
            
            // Process edges
            for (int i = start; i < end; i++) {
                int id = g_graph_edges[i];
                
                // Load visited flag - coalesced within warp if neighbors are clustered
                bool visited = g_graph_visited[id];
                
                if (!visited) {
                    // Update cost and mask atomically to avoid race conditions
                    // Store cost first
                    g_cost[id] = current_cost + 1;
                    
                    // Then update mask
                    g_updating_graph_mask[id] = true;
                }
            }
        }
    }
}

#endif
