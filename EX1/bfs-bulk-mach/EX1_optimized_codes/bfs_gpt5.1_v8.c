#include <time.h>
#include "../bfs.h"

static double bfs_kernel_time = 0.0;

void reset_bfs_kernel_time(void) { bfs_kernel_time = 0.0; }
double get_bfs_kernel_time(void) { return bfs_kernel_time; }

void bfs(node_t nodes[N_NODES], edge_t edges[N_EDGES],
         node_index_t starting_node, level_t level[N_NODES],
         edge_index_t level_counts[N_LEVELS])
{
  node_index_t n;
  edge_index_t e;
  level_t horizon;
  edge_index_t cnt;
  struct timespec kernel_start, kernel_end;

  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  level[starting_node] = 0;
  level_counts[0] = 1;

  // Preload commonly used constants into locals to aid the compiler
  const level_t max_level = MAX_LEVEL;
  const node_index_t n_nodes = N_NODES;
  const level_t n_levels = N_LEVELS;

  loop_horizons:
  for (horizon = 0; horizon < n_levels; ++horizon) {
    cnt = 0;

    // Add unmarked neighbors of the current horizon to the next horizon
    loop_nodes:
    for (n = 0; n < n_nodes; ++n) {
      if (level[n] == horizon) {
        const edge_index_t tmp_begin = nodes[n].edge_begin;
        const edge_index_t tmp_end   = nodes[n].edge_end;

        loop_neighbors:
        for (e = tmp_begin; e < tmp_end; ++e) {
          const node_index_t tmp_dst = edges[e].dst;

          // Load once, then compare
          if (level[tmp_dst] == max_level) { // Unmarked
            level[tmp_dst] = (level_t)(horizon + 1);
            ++cnt;
          }
        }
      }
    }

    level_counts[horizon + 1] = cnt;
    if (cnt == 0) {
      break;
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  bfs_kernel_time += (kernel_end.tv_sec - kernel_start.tv_sec) +
                     (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
