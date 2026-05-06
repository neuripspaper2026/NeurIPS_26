#ifndef _KERNEL_H_
#define _KERNEL_H_

// Assume Node is defined elsewhere as something like:
// struct Node { int starting; int no_of_edges; };

__global__ void Kernel(Node * __restrict__ g_graph_nodes,
                       const int * __restrict__ g_graph_edges,
                       bool * __restrict__ g_graph_mask,
                       bool * __restrict__ g_updating_graph_mask,
                       bool * __restrict__ g_graph_visited,
                       int * __restrict__ g_cost,
                       int no_of_nodes) {
    // Use standard 2D-style indexing to allow flexible block sizes;
    // MAX_THREADS_PER_BLOCK is assumed to be a multiple of warpSize.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < no_of_nodes && g_graph_mask[tid]) {
        // Clear current frontier flag once per active node
        g_graph_mask[tid] = false;

        // Cache per-node values in registers to reduce repeated global loads
        const int edge_start = g_graph_nodes[tid].starting;
        const int edge_count = g_graph_nodes[tid].no_of_edges;
        const int edge_end   = edge_start + edge_count;
        const int base_cost  = g_cost[tid] + 1;

        // Use local loop variables with explicit restrict-like semantics
        // to help the compiler generate better code.
        #pragma unroll 1
        for (int e = edge_start; e < edge_end; ++e) {
            int id = g_graph_edges[e];

            // Load visited flag once; avoid re-reading on store path
            bool visited = g_graph_visited[id];
            if (!visited) {
                g_cost[id] = base_cost;
                g_updating_graph_mask[id] = true;
            }
        }
    }
}

#endif
