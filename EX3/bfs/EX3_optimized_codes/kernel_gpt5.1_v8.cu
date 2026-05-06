#ifndef _KERNEL_H_
#define _KERNEL_H_

// Assume Node is defined elsewhere as:
// struct Node {
//     int starting;
//     int no_of_edges;
// };

#ifndef MAX_THREADS_PER_BLOCK
#define MAX_THREADS_PER_BLOCK 1024
#endif

__global__ void Kernel(Node *g_graph_nodes, int *g_graph_edges,
                       bool *g_graph_mask, bool *g_updating_graph_mask,
                       bool *g_graph_visited, int *g_cost, int no_of_nodes) {
    // Use native CUDA indexing to avoid dependence on MAX_THREADS_PER_BLOCK
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < no_of_nodes && g_graph_mask[tid]) {
        // Clear mask early to avoid re-processing this node
        g_graph_mask[tid] = false;

        // Cache node descriptor in registers
        const Node node = g_graph_nodes[tid];
        const int start  = node.starting;
        const int end    = start + node.no_of_edges;

        // Loop over all outgoing edges of this node
        // Use simple loop to allow compiler unrolling / optimization
        for (int i = start; i < end; ++i) {
            const int id = g_graph_edges[i];

            // Read visited flag once
            const bool visited = g_graph_visited[id];

            // Only update if not visited
            if (!visited) {
                // Use register for source cost to reduce memory traffic
                const int src_cost = g_cost[tid];
                g_cost[id] = src_cost + 1;

                // Mark for next frontier
                g_updating_graph_mask[id] = true;
            }
        }
    }
}

#endif
