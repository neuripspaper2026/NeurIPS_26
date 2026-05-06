#ifndef _KERNEL2_H_
#define _KERNEL2_H_

__global__ void Kernel2(bool *g_graph_mask, bool *g_updating_graph_mask,
                        bool *g_graph_visited, bool *g_over, int no_of_nodes) {
    int tid = blockIdx.x * MAX_THREADS_PER_BLOCK + threadIdx.x;
    
    if (tid < no_of_nodes) {
        // Coalesced read of updating mask
        bool updating = g_updating_graph_mask[tid];
        
        if (updating) {
            // Perform all writes with coalesced access pattern
            g_graph_mask[tid] = true;
            g_graph_visited[tid] = true;
            g_updating_graph_mask[tid] = false;
            
            // Use atomic operation for global flag to ensure correctness
            // This is more efficient than direct write on A100
            atomicExch((int*)g_over, 1);
        }
    }
}

// Warp-optimized version using ballot for better A100 utilization
__global__ void Kernel2_Warp_Optimized(bool *g_graph_mask, bool *g_updating_graph_mask,
                        bool *g_graph_visited, bool *g_over, int no_of_nodes) {
    int tid = blockIdx.x * MAX_THREADS_PER_BLOCK + threadIdx.x;
    
    // Warp-level optimization: check if any thread in warp needs to update
    bool updating = false;
    if (tid < no_of_nodes) {
        updating = g_updating_graph_mask[tid];
    }
    
    // Use warp ballot to detect if any thread in warp has work
    unsigned int warp_mask = __ballot_sync(0xFFFFFFFF, updating);
    
    if (warp_mask != 0) {
        if (updating) {
            // Coalesced memory writes
            g_graph_mask[tid] = true;
            g_graph_visited[tid] = true;
            g_updating_graph_mask[tid] = false;
            
            // Single atomic per warp instead of per thread
            if (__popc(warp_mask) > 0 && (threadIdx.x & 31) == __ffs(warp_mask) - 1) {
                atomicExch((int*)g_over, 1);
            }
        }
    }
}

// Vectorized version for better memory throughput on A100
__global__ void Kernel2_Vectorized(bool *g_graph_mask, bool *g_updating_graph_mask,
                        bool *g_graph_visited, bool *g_over, int no_of_nodes) {
    int tid = blockIdx.x * MAX_THREADS_PER_BLOCK + threadIdx.x;
    
    // Process multiple elements per thread for better memory bandwidth utilization
    const int ELEMENTS_PER_THREAD = 4;
    int base_tid = tid * ELEMENTS_PER_THREAD;
    
    bool any_updated = false;
    
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int current_tid = base_tid + i;
        if (current_tid < no_of_nodes) {
            bool updating = g_updating_graph_mask[current_tid];
            
            if (updating) {
                g_graph_mask[current_tid] = true;
                g_graph_visited[current_tid] = true;
                g_updating_graph_mask[current_tid] = false;
                any_updated = true;
            }
        }
    }
    
    // Reduce atomic contention by using warp-level reduction
    unsigned int warp_updated = __ballot_sync(0xFFFFFFFF, any_updated);
    if (warp_updated != 0 && (threadIdx.x & 31) == 0) {
        atomicExch((int*)g_over, 1);
    }
}

#endif
