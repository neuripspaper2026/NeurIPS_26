#ifndef _KERNEL2_H_
#define _KERNEL2_H_

__global__ void Kernel2(bool *g_graph_mask, bool *g_updating_graph_mask,
                        bool *g_graph_visited, bool *g_over, int no_of_nodes) {
    int tid = blockIdx.x * MAX_THREADS_PER_BLOCK + threadIdx.x;
    
    if (tid < no_of_nodes) {
        // Load updating mask into register to avoid multiple global memory accesses
        bool updating = g_updating_graph_mask[tid];
        
        if (updating) {
            // Perform all writes
            g_graph_mask[tid] = true;
            g_graph_visited[tid] = true;
            g_updating_graph_mask[tid] = false;
            
            // Use atomicExch instead of direct assignment for g_over
            // This ensures proper memory ordering and visibility across threads
            atomicExch((int*)g_over, 1);
        }
    }
}

#endif
