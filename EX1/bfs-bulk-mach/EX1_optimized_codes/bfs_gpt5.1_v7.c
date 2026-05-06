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

  // Initialize levels and level_counts
  {
    node_index_t i;
    for (i = 0; i < N_NODES; ++i) {
      level[i] = MAX_LEVEL;
    }
  }
  {
    level_t i;
    for (i = 0; i < N_LEVELS; ++i) {
      level_counts[i] = 0;
    }
  }

  level[starting_node] = 0;
  level_counts[0] = 1;

  for (level_t horizon = 0; horizon < N_LEVELS - 1; ++horizon) {
    edge_index_t cnt = 0;

    // Skip this horizon if no nodes were discovered at this level
    if (level_counts[horizon] == 0)
      break;

    for (node_index_t n = 0; n < N_NODES; ++n) {
      if (level[n] == horizon) {
        edge_index_t tmp_begin = nodes[n].edge_begin;
        edge_index_t tmp_end = nodes[n].edge_end;

        for (edge_index_t e = tmp_begin; e < tmp_end; ++e) {
          node_index_t tmp_dst = edges[e].dst;

          // Only process previously unvisited nodes
          if (level[tmp_dst] == MAX_LEVEL) {
            level[tmp_dst] = horizon + 1;
            ++cnt;
          }
        }
      }
    }

    level_counts[horizon + 1] = cnt;
    if (cnt == 0)
      break;
  }

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  bfs_kernel_time += (kernel_end.tv_sec - kernel_start.tv_sec) +
                     (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
