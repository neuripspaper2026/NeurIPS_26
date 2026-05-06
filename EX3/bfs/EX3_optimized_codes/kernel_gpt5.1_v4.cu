#ifndef _KERNEL_H_
#define _KERNEL_H_

// Assuming Node and MAX_THREADS_PER_BLOCK are defined elsewhere as in the original project.

// Optimized BFS-like kernel for NVIDIA A100
__global__ void Kernel(Node * __restrict__ g_graph_nodes,
                       const int * __restrict__ g_graph_edges,
                       bool * __restrict__ g_graph_mask,
                       bool * __restrict__ g_updating_graph_mask,
                       bool * __restrict__ g_graph_visited,
                       int * __restrict__ g_cost,
                       const int no_of_nodes) {
    // Use a grid-stride loop to better utilize SMs for large graphs
    const int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    const int totalThreads   = gridDim.x * blockDim.x;

    // Process multiple nodes per thread if necessary
    for (int tid = globalThreadId; tid < no_of_nodes; tid += totalThreads) {

        // Read mask once and early-out for inactive nodes
        if (!g_graph_mask[tid]) {
            continue;
        }

        // Clear mask (this thread is processing this frontier node)
        g_graph_mask[tid] = false;

        // Cache node data in registers
        const int edge_start = g_graph_nodes[tid].starting;
        const int edge_end   = edge_start + g_graph_nodes[tid].no_of_edges;

        // Neighbor iteration
        for (int i = edge_start; i < edge_end; ++i) {
            const int id = g_graph_edges[i];

            // Opportunistic check: reduce redundant writes
            if (!g_graph_visited[id]) {
                // Update cost and next frontier mask
                g_cost[id] = g_cost[tid] + 1;
                g_updating_graph_mask[id] = true;
            }
        }
    }
}

#endif
