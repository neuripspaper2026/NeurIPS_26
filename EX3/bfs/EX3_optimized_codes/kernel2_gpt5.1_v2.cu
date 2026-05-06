#ifndef _KERNEL2_H_
#define _KERNEL2_H_

// Tailored for NVIDIA A100 GPUs; assumes MAX_THREADS_PER_BLOCK is defined elsewhere.

__global__ void Kernel2(bool * __restrict__ g_graph_mask,
                        bool * __restrict__ g_updating_graph_mask,
                        bool * __restrict__ g_graph_visited,
                        bool * __restrict__ g_over,
                        int no_of_nodes) {
    int tid = blockIdx.x * MAX_THREADS_PER_BLOCK + threadIdx.x;

    // Early exit if out of bounds
    if (tid >= no_of_nodes) {
        return;
    }

    // Read once into a register to avoid repeated global memory access
    const bool updating = g_updating_graph_mask[tid];

    // Most threads will typically be inactive; early exit to reduce divergence work
    if (!updating) {
        return;
    }

    // Update frontier/bookkeeping arrays; accesses are naturally coalesced
    g_graph_mask[tid]         = true;
    g_graph_visited[tid]      = true;
    g_updating_graph_mask[tid] = false;

    // Warp-aggregated flag update to reduce contention on g_over
    // Any lane with 'updating == true' will cause one lane per warp to set g_over.
    unsigned int mask = __ballot_sync(0xffffffff, updating);
    if (mask) {
        // Use lane 0 in the warp as the designated writer when any thread is active
        if ((threadIdx.x & 31) == 0) {
            *g_over = true;
        }
    }
}

#endif
