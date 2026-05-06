#ifndef _KERNEL2_H_
#define _KERNEL2_H_

// Use launch_bounds consistent with MAX_THREADS_PER_BLOCK and other kernels
__global__ __launch_bounds__(MAX_THREADS_PER_BLOCK, 2)
void Kernel2(bool * __restrict__ g_graph_mask,
             bool * __restrict__ g_updating_graph_mask,
             bool * __restrict__ g_graph_visited,
             bool * __restrict__ g_over,
             int                no_of_nodes) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= no_of_nodes) {
        return;
    }

    // Cache updating mask value in a register to reduce global memory traffic
    bool updating = g_updating_graph_mask[tid];
    if (!updating) {
        return;
    }

    // Mark the node as active and visited; keep accesses coalesced
    g_graph_mask[tid]    = true;
    g_graph_visited[tid] = true;

    // At least one thread found an update; avoid branching by writing directly
    // Multiple threads may write 'true' concurrently; this is benign
    *g_over = true;

    // Clear updating mask
    g_updating_graph_mask[tid] = false;
}

#endif
