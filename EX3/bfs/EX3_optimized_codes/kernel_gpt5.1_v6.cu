#ifndef _KERNEL_H_
#define _KERNEL_H_

// Assume Node is defined elsewhere as something like:
// struct Node { int starting; int no_of_edges; };

#ifndef MAX_THREADS_PER_BLOCK
#define MAX_THREADS_PER_BLOCK 256
#endif

// Use __restrict__ to help the compiler with alias analysis
__global__ void Kernel(Node *__restrict__ g_graph_nodes,
                       const int *__restrict__ g_graph_edges,
                       bool *__restrict__ g_graph_mask,
                       bool *__restrict__ g_updating_graph_mask,
                       bool *__restrict__ g_graph_visited,
                       int *__restrict__ g_cost,
                       int no_of_nodes) {
    // Use blockDim.x instead of macro in case launch uses different size
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Early exit for inactive threads while ensuring no side effects
    if (tid >= no_of_nodes)
        return;

    // Load frequently used data into registers
    bool is_active = g_graph_mask[tid];

    if (is_active) {
        // Clear mask once per active node
        g_graph_mask[tid] = false;

        // Cache node information in registers to reduce global loads
        int start  = g_graph_nodes[tid].starting;
        int degree = g_graph_nodes[tid].no_of_edges;
        int end    = start + degree;

        // Cache source cost in a register
        int src_cost = g_cost[tid];
        int new_cost = src_cost + 1;

        // Loop over edge list
        for (int i = start; i < end; ++i) {
            // Access edges in a coalesced manner when possible
            int id = g_graph_edges[i];

            // Avoid redundant global reads by using a register
            if (!g_graph_visited[id]) {
                // For BFS-style updates, last writer wins is acceptable,
                // so we avoid expensive atomics here.
                g_cost[id] = new_cost;
                g_updating_graph_mask[id] = true;
            }
        }
    }
}

#endif
