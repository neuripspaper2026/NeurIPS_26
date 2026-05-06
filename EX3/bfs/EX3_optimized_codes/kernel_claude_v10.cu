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
            // Mark as not in mask (coalesced write)
            g_graph_mask[tid] = false;
            
            // Load node data once
            Node node = g_graph_nodes[tid];
            int cost_tid = g_cost[tid];
            int new_cost = cost_tid + 1;
            
            int start = node.starting;
            int end = start + node.no_of_edges;
            
            // Process edges
            for (int i = start; i < end; i++) {
                int id = g_graph_edges[i];
                
                // Load visited status once
                bool visited = g_graph_visited[id];
                
                if (!visited) {
                    // Coalesced writes when possible
                    g_cost[id] = new_cost;
                    g_updating_graph_mask[id] = true;
                }
            }
        }
    }
}

#endif
