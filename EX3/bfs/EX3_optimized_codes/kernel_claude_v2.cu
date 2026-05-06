#ifndef _KERNEL_H_
#define _KERNEL_H_

__global__ void Kernel(Node *g_graph_nodes, int *g_graph_edges,
                       bool *g_graph_mask, bool *g_updating_graph_mask,
                       bool *g_graph_visited, int *g_cost, int no_of_nodes) {
    int tid = blockIdx.x * MAX_THREADS_PER_BLOCK + threadIdx.x;
    
    if (tid < no_of_nodes) {
        // Load graph mask into register with relaxed memory ordering
        bool mask = __ldg(&g_graph_mask[tid]);
        
        if (mask) {
            // Store false to graph mask
            g_graph_mask[tid] = false;
            
            // Load node information into registers
            int starting = g_graph_nodes[tid].starting;
            int no_of_edges = g_graph_nodes[tid].no_of_edges;
            int cost_tid = __ldg(&g_cost[tid]);
            int new_cost = cost_tid + 1;
            
            // Process edges with vectorized loads where possible
            int edge_end = starting + no_of_edges;
            
            // Unroll loop for better instruction-level parallelism
            #pragma unroll 4
            for (int i = starting; i < edge_end; i++) {
                int id = __ldg(&g_graph_edges[i]);
                
                // Use read-only cache for visited array
                bool visited = __ldg(&g_graph_visited[id]);
                
                if (!visited) {
                    g_cost[id] = new_cost;
                    g_updating_graph_mask[id] = true;
                }
            }
        }
    }
}

#endif
