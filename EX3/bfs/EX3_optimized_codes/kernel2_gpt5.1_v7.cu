#ifndef _KERNEL2_H_
#define _KERNEL2_H_

__global__ void Kernel2(bool * __restrict__ g_graph_mask,
                        bool * __restrict__ g_updating_graph_mask,
                        bool * __restrict__ g_graph_visited,
                        bool * __restrict__ g_over,
                        int                 no_of_nodes) {
    // Use blockDim.x for consistency and flexibility
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    bool in_range = (tid < no_of_nodes);
    bool will_update = false;

    if (in_range) {
        will_update = g_updating_graph_mask[tid];
    }

    // Warp-level check: if no thread in the warp will update, all can exit
    unsigned int update_mask = __ballot_sync(0xffffffff, in_range && will_update);
    if (update_mask == 0) {
        return;
    }

    if (in_range && will_update) {
        g_graph_mask[tid]        = true;
        g_graph_visited[tid]     = true;
        g_updating_graph_mask[tid] = false;

        // Only one thread per warp sets *g_over to reduce contention
        // Pick the first active lane in this warp
        int leader_lane = __ffs(update_mask) - 1; // __ffs is 1-based
        if ((threadIdx.x & 31) == leader_lane) {
            *g_over = true;
        }
    }
}

#endif
