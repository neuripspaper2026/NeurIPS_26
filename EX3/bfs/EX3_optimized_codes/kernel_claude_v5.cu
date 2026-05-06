#ifndef _KERNEL_H_
#define _KERNEL_H_

__global__ void Kernel(Node *g_graph_nodes, int *g_graph_edges,
                       bool *g_graph_mask, bool *g_updating_graph_mask,
                       bool *g_graph_visited, int *g_cost, int no_of_nodes) {
    int tid = blockIdx.x * MAX_THREADS_PER_BLOCK + threadIdx.x;
    
    if (tid < no_of_nodes) {
        // Load graph mask value once and use register
        bool is_masked = g_graph_mask[tid];
        
        if (is_masked) {
            g_graph_mask[tid] = false;
            
            // Cache node data in registers
            Node node = g_graph_nodes[tid];
            int start = node.starting;
            int end = start + node.no_of_edges;
            int current_cost = g_cost[tid];
            int next_cost = current_cost + 1;
            
            // Process edges with improved memory access patterns
            for (int i = start; i < end; i++) {
                int id = g_graph_edges[i];
                
                // Load visited status once
                bool is_visited = g_graph_visited[id];
                
                if (!is_visited) {
                    g_cost[id] = next_cost;
                    g_updating_graph_mask[id] = true;
                }
            }
        }
    }
}

#endif
