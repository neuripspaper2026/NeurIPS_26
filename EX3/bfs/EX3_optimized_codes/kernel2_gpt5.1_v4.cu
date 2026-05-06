#ifndef _KERNEL2_H_
#define _KERNEL2_H_

__global__ void Kernel2(bool * __restrict__ g_graph_mask,
                        bool * __restrict__ g_updating_graph_mask,
                        bool * __restrict__ g_graph_visited,
                        bool * __restrict__ g_over,
                        const int no_of_nodes) {
    // Use standard CUDA thread indexing for flexibility with different block sizes
    const int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    const int totalThreads   = gridDim.x * blockDim.x;

    // Grid-stride loop for better GPU utilization on large graphs
    for (int tid = globalThreadId; tid < no_of_nodes; tid += totalThreads) {
        // Load mask once into a register
        const bool updating_mask = g_updating_graph_mask[tid];

        if (!updating_mask) {
            continue;
        }

        // Update node state for the next BFS frontier
        g_graph_mask[tid]         = true;
        g_graph_visited[tid]      = true;
        g_updating_graph_mask[tid] = false;

        // Set the over flag if any thread performs an update.
        // Multiple concurrent writes of 'true' are benign (idempotent).
        *g_over = true;
    }
}

#endif
