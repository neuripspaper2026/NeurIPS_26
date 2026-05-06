#ifndef _KERNEL2_H_
#define _KERNEL2_H_

__global__ void Kernel2(bool * __restrict__ g_graph_mask,
                        bool * __restrict__ g_updating_graph_mask,
                        bool * __restrict__ g_graph_visited,
                        bool * __restrict__ g_over,
                        int                no_of_nodes) {
    // Use standard CUDA thread indexing for flexible launch configuration.
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= no_of_nodes) return;

    // Load once from global memory.
    const bool updating = g_updating_graph_mask[tid];

    if (!updating) return;

    // Update masks and visited flags.
    g_graph_mask[tid]        = true;
    g_graph_visited[tid]     = true;
    g_updating_graph_mask[tid] = false;

    // Warp-aggregated flag update to reduce contention on g_over.
    // If any thread in the warp performed an update, a single lane sets g_over.
    unsigned int mask = __activemask();
    int any_updated   = __any_sync(mask, updating);

    if (any_updated) {
        // Designate one lane per warp (lane 0) to set the global flag.
        if ((threadIdx.x & 31) == 0) {
            *g_over = true;
        }
    }
}

#endif
