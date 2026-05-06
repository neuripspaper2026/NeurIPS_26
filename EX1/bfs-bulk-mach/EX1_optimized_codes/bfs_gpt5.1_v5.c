#include <time.h>
#include "../bfs.h"

static double bfs_kernel_time = 0.0;

void reset_bfs_kernel_time(void) { bfs_kernel_time = 0.0; }
double get_bfs_kernel_time(void) { return bfs_kernel_time; }

void bfs(node_t nodes[N_NODES], edge_t edges[N_EDGES],
         node_index_t starting_node, level_t level[N_NODES],
         edge_index_t level_counts[N_LEVELS])
{
  struct timespec kernel_start, kernel_end;

  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  // Initialize BFS
  level[starting_node] = 0;
  level_counts[0] = 1;

  // Use arrays as frontiers instead of scanning all nodes every level
  static node_index_t frontier[N_NODES];
  static node_index_t next_frontier[N_NODES];

  node_index_t frontier_size = 1;
  node_index_t next_frontier_size = 0;
  frontier[0] = starting_node;

  level_t horizon = 0;

  while (horizon < N_LEVELS - 1 && frontier_size > 0) {
    next_frontier_size = 0;

    for (node_index_t i = 0; i < frontier_size; ++i) {
      node_index_t n = frontier[i];
      edge_index_t tmp_begin = nodes[n].edge_begin;
      edge_index_t tmp_end = nodes[n].edge_end;

      for (edge_index_t e = tmp_begin; e < tmp_end; ++e) {
        node_index_t tmp_dst = edges[e].dst;

        if (level[tmp_dst] == MAX_LEVEL) { // Unmarked
          level[tmp_dst] = horizon + 1;
          next_frontier[next_frontier_size++] = tmp_dst;
        }
      }
    }

    level_counts[horizon + 1] = next_frontier_size;
    if (next_frontier_size == 0)
      break;

    // Swap frontiers
    node_index_t *tmp_ptr = frontier;
    frontier = next_frontier;
    next_frontier = tmp_ptr;

    frontier_size = next_frontier_size;
    ++horizon;
  }

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  bfs_kernel_time += (kernel_end.tv_sec - kernel_start.tv_sec) +
                     (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
