#ifndef _KERNEL2_H_
#define _KERNEL2_H_

#ifndef MAX_THREADS_PER_BLOCK
#define MAX_THREADS_PER_BLOCK 256
#endif

__global__ void Kernel2(bool *__restrict__ g_graph_mask,
                        bool *__restrict__ g_updating_graph_mask,
                        bool *__restrict__ g_graph_visited,
                        bool *__restrict__ g_over,
                        int no_of_nodes) {
    // Match launch configuration flexibility using blockDim.x
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < no_of_nodes) {
        // Cache flag in a register to avoid multiple global reads
        bool updating = g_updating_graph_mask[tid];

        if (updating) {
            g_graph_mask[tid]       = true;
            g_graph_visited[tid]    = true;
            g_updating_graph_mask[tid] = false;

            // Warp-aggregated write to g_over to avoid redundant stores
            // Any thread with updating == true in the warp will set lane_mask != 0
#if __CUDA_ARCH__ >= 700
            unsigned int mask = __activemask();
            unsigned int lane_mask = __ballot_sync(mask, updating);
            // Only one lane per warp performs the global store if any bit is set
            if (lane_mask && ((threadIdx.x & (warpSize - 1)) == 0)) {
                *g_over = true;
            }
#else
            // Fallback for older architectures – still correct
            *g_over = true;
#endif
        }
    }
}

#endif
