#ifndef _KERNEL_H_
#define _KERNEL_H_

__global__ void Kernel(Node *g_graph_nodes, int *g_graph_edges,
                       bool *g_graph_mask, bool *g_updating_graph_mask,
                       bool *g_graph_visited, int *g_cost, int no_of_nodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < no_of_nodes && g_graph_mask[tid]) {
        g_graph_mask[tid] = false;
        Node node = g_graph_nodes[tid];
        int start = node.starting;
        int end = node.no_of_edges + start;
        
        for (int i = start; i < end; i++) {
            int id = g_graph_edges[i];
            if (!g_graph_visited[id]) {
                g_cost[id] = g_cost[tid] + 1;
                g_updating_graph_mask[id] = true;
            }
        }
    }
}

#endif
