#ifndef _KERNEL_H_
#define _KERNEL_H_

// Assume Node is defined elsewhere, as in original code.
// Optimization flags:
// - Use __restrict__ to help compiler with alias analysis
// - Use __ldg for read-only data to leverage read-only cache on A100
// - Use __launch_bounds__ to guide occupancy for A100

#ifndef MAX_THREADS_PER_BLOCK
#define MAX_THREADS_PER_BLOCK 256
#endif

__global__ __launch_bounds__(MAX_THREADS_PER_BLOCK, 2)
void Kernel(Node *__restrict__ g_graph_nodes,
            const int *__restrict__ g_graph_edges,
            bool *__restrict__ g_graph_mask,
            bool *__restrict__ g_updating_graph_mask,
            bool *__restrict__ g_graph_visited,
            int *__restrict__ g_cost,
            int no_of_nodes) {
    int tid = blockIdx.x * MAX_THREADS_PER_BLOCK + threadIdx.x;

    if (tid < no_of_nodes) {
        // Load mask once and early exit if not active
        bool active = g_graph_mask[tid];
        if (active) {
            // Clear mask once
            g_graph_mask[tid] = false;

            // Cache frequently used per-node data in registers
            int start = __ldg(&g_graph_nodes[tid].starting);
            int edge_count = __ldg(&g_graph_nodes[tid].no_of_edges);
            int base_cost = g_cost[tid] + 1;

            int end = start + edge_count;

#pragma unroll 4
            for (int i = start; i < end; ++i) {
                int id = __ldg(&g_graph_edges[i]);
                // Reduce global loads with __ldg for visited array
                if (!__ldg(&g_graph_visited[id])) {
                    g_cost[id] = base_cost;
                    g_updating_graph_mask[id] = true;
                }
            }
        }
    }
}

#endif
