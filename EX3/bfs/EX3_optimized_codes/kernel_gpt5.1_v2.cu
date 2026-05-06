#ifndef _KERNEL_H_
#define _KERNEL_H_

// Assuming Node definition and MAX_THREADS_PER_BLOCK are provided elsewhere.
// This optimized kernel is tailored for NVIDIA A100 GPUs.

__global__ void Kernel(Node * __restrict__ g_graph_nodes,
                       const int * __restrict__ g_graph_edges,
                       bool * __restrict__ g_graph_mask,
                       bool * __restrict__ g_updating_graph_mask,
                       bool * __restrict__ g_graph_visited,
                       int * __restrict__ g_cost,
                       int no_of_nodes) {
    int tid = blockIdx.x * MAX_THREADS_PER_BLOCK + threadIdx.x;

    if (tid >= no_of_nodes) {
        return;
    }

    // Early exit if this node is not active in current frontier
    if (!g_graph_mask[tid]) {
        return;
    }

    // Clear frontier flag for this node
    g_graph_mask[tid] = false;

    // Cache node descriptor in registers
    const int start_edge = g_graph_nodes[tid].starting;
    const int edge_count = g_graph_nodes[tid].no_of_edges;
    const int end_edge   = start_edge + edge_count;

    // Cache cost in register
    const int base_cost = g_cost[tid];
    const int new_cost  = base_cost + 1;

    // Process neighbor list
    // Keep loop simple to allow compiler unrolling and better ILP
    for (int edge_idx = start_edge; edge_idx < end_edge; ++edge_idx) {
        const int id = g_graph_edges[edge_idx];

        // Avoid unnecessary writes if already visited
        if (!g_graph_visited[id]) {
            g_cost[id] = new_cost;
            g_updating_graph_mask[id] = true;
        }
    }
}

#endif
