#ifndef _KERNEL2_H_
#define _KERNEL2_H_

#ifndef MAX_THREADS_PER_BLOCK
#define MAX_THREADS_PER_BLOCK 256
#endif

// Use launch bounds to guide occupancy and align with kernel.cu
__global__ __launch_bounds__(MAX_THREADS_PER_BLOCK, 2)
void Kernel2(bool *__restrict__ g_graph_mask,
             bool *__restrict__ g_updating_graph_mask,
             bool *__restrict__ g_graph_visited,
             bool *__restrict__ g_over,
             int no_of_nodes) {
    int tid = blockIdx.x * MAX_THREADS_PER_BLOCK + threadIdx.x;

    if (tid < no_of_nodes) {
        // Load flag once into a register to avoid multiple global reads
        bool updating = g_updating_graph_mask[tid];

        if (updating) {
            g_graph_mask[tid]        = true;
            g_graph_visited[tid]     = true;
            g_updating_graph_mask[tid] = false;

            // Single thread per block updates the global convergence flag
            // to reduce contention on atomics across all threads
            if (threadIdx.x == 0) {
                // Any write of 'true' is logically ORed via atomic
                atomicExch(g_over, true);
            }
        }
    }
}

#endif
