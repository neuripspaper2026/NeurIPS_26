#ifndef _KERNEL_H_
#define _KERNEL_H_

// Assume Node is defined elsewhere, as in the original context.
// Example (do NOT duplicate if already defined in another header):
// struct Node {
//     int starting;
//     int no_of_edges;
// };

__global__ void Kernel(Node * __restrict__ g_graph_nodes,
                       int  * __restrict__ g_graph_edges,
                       bool * __restrict__ g_graph_mask,
                       bool * __restrict__ g_updating_graph_mask,
                       bool * __restrict__ g_graph_visited,
                       int  * __restrict__ g_cost,
                       int no_of_nodes) {
    // Use native CUDA blockDim.x for flexibility while preserving original mapping
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Early exit if out of bounds or inactive
    if (tid >= no_of_nodes || !g_graph_mask[tid]) {
        return;
    }

    // Clear mask for this node once we know we will process it
    g_graph_mask[tid] = false;

    // Cache node data in registers to avoid repeated global loads
    const Node node      = g_graph_nodes[tid];
    const int edge_start = node.starting;
    const int edge_end   = edge_start + node.no_of_edges;

    // Cache source cost in register
    const int src_cost = g_cost[tid];

    // Iterate neighbors; use local temporaries and pointers for better codegen
    int       i   = edge_start;
    int * __restrict__ edges_ptr  = g_graph_edges;
    bool * __restrict__ visited   = g_graph_visited;
    bool * __restrict__ upd_mask  = g_updating_graph_mask;
    int  * __restrict__ cost_ptr  = g_cost;

    // Unroll small loops to reduce loop overhead; factor chosen conservatively
    #pragma unroll 4
    for (; i < edge_end; ++i) {
        int id = edges_ptr[i];

        // Read-mostly visited flag; branch is likely divergent but unavoidable
        if (!visited[id]) {
            // Write cost and mark for next frontier
            cost_ptr[id] = src_cost + 1;
            upd_mask[id] = true;
        }
    }
}

#endif
