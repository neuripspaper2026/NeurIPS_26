#ifndef _KERNEL_H_
#define _KERNEL_H_

// Assume Node definition and MAX_THREADS_PER_BLOCK are provided elsewhere.

__global__ void Kernel(Node * __restrict__ g_graph_nodes,
                       int   * __restrict__ g_graph_edges,
                       bool  * __restrict__ g_graph_mask,
                       bool  * __restrict__ g_updating_graph_mask,
                       bool  * __restrict__ g_graph_visited,
                       int   * __restrict__ g_cost,
                       int               no_of_nodes) {
    // Use native thread/block indices to allow flexible launch configs.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Early exit for out-of-range threads.
    if (tid >= no_of_nodes) return;

    // Load mask once per thread; avoid repeated global loads.
    bool active = g_graph_mask[tid];
    if (!active) return;

    // Clear mask once we know this node is being processed.
    g_graph_mask[tid] = false;

    // Cache node metadata in registers to reduce global memory traffic.
    const int start = g_graph_nodes[tid].starting;
    const int end   = start + g_graph_nodes[tid].no_of_edges;

    // Cache source cost in a register to avoid reloading each iteration.
    const int src_cost = g_cost[tid] + 1;

    // Iterate over adjacency list.
    // Memory access to g_graph_edges is already naturally coalesced
    // when frontiers contain consecutive vertices.
#pragma unroll 4
    for (int i = start; i < end; ++i) {
        const int id = g_graph_edges[i];

        // Load visited flag once; use branch to skip already-visited nodes.
        if (!g_graph_visited[id]) {
            // Write cost and update mask.
            g_cost[id]                 = src_cost;
            g_updating_graph_mask[id]  = true;
        }
    }
}

#endif
