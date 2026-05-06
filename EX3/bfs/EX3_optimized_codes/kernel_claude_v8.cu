#ifndef _KERNEL_H_
#define _KERNEL_H_

__global__ void Kernel(Node *g_graph_nodes, int *g_graph_edges,
                       bool *g_graph_mask, bool *g_updating_graph_mask,
                       bool *g_graph_visited, int *g_cost, int no_of_nodes) {
    int tid = blockIdx.x * MAX_THREADS_PER_BLOCK + threadIdx.x;
    
    if (tid < no_of_nodes) {
        // Load graph mask value once and store in register
        bool mask_value = g_graph_mask[tid];
        
        if (mask_value) {
            // Mark as not in mask immediately
            g_graph_mask[tid] = false;
            
            // Load node information into registers
            int starting = g_graph_nodes[tid].starting;
            int no_of_edges = g_graph_nodes[tid].no_of_edges;
            int cost_tid = g_cost[tid];
            int new_cost = cost_tid + 1;
            
            // Process edges with better memory access patterns
            for (int i = starting; i < (starting + no_of_edges); i++) {
                int id = g_graph_edges[i];
                
                // Use single read for visited check
                bool is_visited = g_graph_visited[id];
                
                if (!is_visited) {
                    // Coalesced writes when possible
                    g_cost[id] = new_cost;
                    g_updating_graph_mask[id] = true;
                }
            }
        }
    }
}

#endif
