#ifndef _KERNEL_H_
#define _KERNEL_H_

// Assuming Node is defined elsewhere, and MAX_THREADS_PER_BLOCK is a compile-time constant

// Use launch_bounds to help the compiler optimize for the chosen block size
__global__ __launch_bounds__(MAX_THREADS_PER_BLOCK, 2)
void Kernel(Node * __restrict__ g_graph_nodes,
            int   * __restrict__ g_graph_edges,
            bool  * __restrict__ g_graph_mask,
            bool  * __restrict__ g_updating_graph_mask,
            bool  * __restrict__ g_graph_visited,
            int   * __restrict__ g_cost,
            int              no_of_nodes) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Early exit if thread is out of range; keeps control flow simple and avoids extra work
    if (global_tid >= no_of_nodes) return;

    // Cache mask value in register to reduce global memory traffic
    bool is_active = g_graph_mask[global_tid];
    if (!is_active) return;

    // Clear mask once per active vertex
    g_graph_mask[global_tid] = false;

    // Cache node data in registers
    const int start_edge = g_graph_nodes[global_tid].starting;
    const int edge_count = g_graph_nodes[global_tid].no_of_edges;
    const int end_edge   = start_edge + edge_count;
    const int base_cost  = g_cost[global_tid] + 1;

    // Stride through edges with warp-synchronous access pattern to improve memory coalescing
    const int lane_id = threadIdx.x & 31;
    int e = start_edge + lane_id;

    // Process all outgoing edges; threads in a warp traverse neighbors in a strided, coalesced way
#pragma unroll 2
    for (; e < end_edge; e += 32) {
        int id = g_graph_edges[e];

        // Use a simple check; no atomics are introduced to preserve original semantics for this context
        if (!g_graph_visited[id]) {
            g_cost[id] = base_cost;
            g_updating_graph_mask[id] = true;
        }
    }
}

#endif
