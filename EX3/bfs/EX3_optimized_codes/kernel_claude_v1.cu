#ifndef _KERNEL_H_
#define _KERNEL_H_

__global__ void Kernel(Node *g_graph_nodes, int *g_graph_edges,
                       bool *g_graph_mask, bool *g_updating_graph_mask,
                       bool *g_graph_visited, int *g_cost, int no_of_nodes) {
    int tid = blockIdx.x * MAX_THREADS_PER_BLOCK + threadIdx.x;
    
    if (tid < no_of_nodes) {
        // Load graph mask into register to avoid redundant global memory access
        bool mask = g_graph_mask[tid];
        
        if (mask) {
            // Write mask update once
            g_graph_mask[tid] = false;
            
            // Load node data into registers for better cache utilization
            Node node = g_graph_nodes[tid];
            int start = node.starting;
            int end = start + node.no_of_edges;
            int cost_tid = g_cost[tid];
            int new_cost = cost_tid + 1;
            
            // Process edges with reduced memory traffic
            for (int i = start; i < end; i++) {
                int id = g_graph_edges[i];
                
                // Load visited flag once
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
