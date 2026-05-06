#ifndef _KERNEL_H_
#define _KERNEL_H_

// Assume Node is defined elsewhere and MAX_THREADS_PER_BLOCK is properly set.

__global__ void Kernel(Node * __restrict__ g_graph_nodes,
                       int   * __restrict__ g_graph_edges,
                       bool  * __restrict__ g_graph_mask,
                       bool  * __restrict__ g_updating_graph_mask,
                       bool  * __restrict__ g_graph_visited,
                       int   * __restrict__ g_cost,
                       int                 no_of_nodes) {
    // Use built-in blockDim.x to avoid relying on a macro for indexing
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Warp-level early exit to reduce divergence and unnecessary work
    bool in_range = (tid < no_of_nodes);
    bool is_active = false;
    if (in_range) {
        is_active = g_graph_mask[tid];
    }

    // Check if any thread in the warp is active; if not, all threads in this warp can skip
    unsigned int active_mask = __ballot_sync(0xffffffff, in_range && is_active);
    if (active_mask == 0) {
        return;
    }

    if (in_range && is_active) {
        // Clear the mask entry once per active node
        g_graph_mask[tid] = false;

        // Cache node info in registers to avoid repeated global memory loads
        int start_edge = g_graph_nodes[tid].starting;
        int end_edge   = start_edge + g_graph_nodes[tid].no_of_edges;

        // Traverse adjacency list
#pragma unroll 4
        for (int i = start_edge; i < end_edge; ++i) {
            int id = g_graph_edges[i];

            // First check without atomics to avoid unnecessary RMW operations
            if (!g_graph_visited[id]) {
                // Use atomicCAS to prevent multiple threads from updating the same node concurrently
                // bool is typically 1 byte; cast to unsigned int pointer for 32-bit atomic operation
                unsigned int *visited_int = reinterpret_cast<unsigned int *>(g_graph_visited) + (id >> 2);
                unsigned int mask = 1u << ((id & 3) * 8);

                unsigned int old_val = atomicOr(visited_int, mask);
                if ((old_val & mask) == 0) {
                    // This thread is the first to visit 'id'
                    g_cost[id] = g_cost[tid] + 1;
                    g_updating_graph_mask[id] = true;
                }
            }
        }
    }
}

#endif
