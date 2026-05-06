#ifndef _KERNEL2_H_
#define _KERNEL2_H_

__global__ void Kernel2(bool * __restrict__ g_graph_mask,
                        bool * __restrict__ g_updating_graph_mask,
                        bool * __restrict__ g_graph_visited,
                        bool * __restrict__ g_over,
                        int no_of_nodes) {
    // Use standard CUDA indexing; assume MAX_THREADS_PER_BLOCK == blockDim.x
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Use a local flag to avoid redundant global loads
    if (tid < no_of_nodes) {
        bool update_flag = g_updating_graph_mask[tid];
        if (update_flag) {
            // Coalesced writes: each thread touches its own index
            g_graph_mask[tid]      = true;
            g_graph_visited[tid]   = true;
            g_updating_graph_mask[tid] = false;

            // At least one thread updated this iteration
            *g_over = true;
        }
    }
}

#endif
