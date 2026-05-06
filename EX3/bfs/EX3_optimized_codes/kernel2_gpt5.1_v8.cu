#ifndef _KERNEL2_H_
#define _KERNEL2_H_

#ifndef MAX_THREADS_PER_BLOCK
#define MAX_THREADS_PER_BLOCK 1024
#endif

__global__ void Kernel2(bool * __restrict__ g_graph_mask,
                        bool * __restrict__ g_updating_graph_mask,
                        bool * __restrict__ g_graph_visited,
                        bool * __restrict__ g_over,
                        int no_of_nodes) {
    // Use native CUDA indexing; keep MAX_THREADS_PER_BLOCK for compatibility
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard against out-of-bounds
    if (tid < no_of_nodes) {
        // Load once into a register to avoid redundant global reads
        const bool updating = g_updating_graph_mask[tid];

        if (updating) {
            // Coalesced writes when frontiers are dense
            g_graph_mask[tid]        = true;
            g_graph_visited[tid]     = true;
            g_updating_graph_mask[tid] = false;

            // Minimize global atomics / writes to g_over using warp-level election.
            // Use atomicOr on the first active lane in the warp only.
            unsigned int active = __ballot_sync(0xFFFFFFFF, updating);
            const int leader_lane = __ffs(active) - 1;  // first set bit -> leader
            const int lane_id = threadIdx.x & 31;

            if (lane_id == leader_lane) {
                // atomicOr ensures correctness even if multiple CTAs write concurrently
                atomicOr(reinterpret_cast<int*>(g_over), 1);
            }
        }
    }
}

#endif
