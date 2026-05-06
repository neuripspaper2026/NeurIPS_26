#ifndef _KERNEL2_H_
#define _KERNEL2_H_

__global__ void Kernel2(bool * __restrict__ g_graph_mask,
                        bool * __restrict__ g_updating_graph_mask,
                        bool * __restrict__ g_graph_visited,
                        bool * __restrict__ g_over,
                        int no_of_nodes) {
    // Use blockDim.x to match launch configuration (see Kernel in kernel.cu)
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check first to avoid invalid memory traffic
    if (tid >= no_of_nodes) {
        return;
    }

    // Load updating flag into a register to avoid multiple global reads
    const bool updating = g_updating_graph_mask[tid];
    if (!updating) {
        return;
    }

    // Cache pointer to single global flag and update through a register temp
    bool * __restrict__ over_ptr = g_over;

    // These stores are naturally coalesced when threads within a warp
    // operate on consecutive tids.
    g_graph_mask[tid]        = true;
    g_graph_visited[tid]     = true;
    *over_ptr                = true;
    g_updating_graph_mask[tid] = false;
}

#endif
