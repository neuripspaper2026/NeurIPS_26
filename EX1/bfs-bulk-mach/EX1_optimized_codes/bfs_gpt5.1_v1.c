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

  // Initialize BFS state
  for (n = 0; n < N_NODES; ++n) {
    level[n] = MAX_LEVEL;
  }
  for (horizon = 0; horizon < N_LEVELS; ++horizon) {
    level_counts[horizon] = 0;
  }

  level[starting_node] = 0;
  level_counts[0] = 1;

  loop_horizons:
  for (horizon = 0; horizon < N_LEVELS - 1; ++horizon) {
    cnt = 0;
    const level_t current_level = horizon;
    const level_t next_level = horizon + 1;

    loop_nodes:
    for (n = 0; n < N_NODES; ++n) {
      if (level[n] == current_level) {
        const edge_index_t tmp_begin = nodes[n].edge_begin;
        const edge_index_t tmp_end   = nodes[n].edge_end;

        loop_neighbors:
        for (e = tmp_begin; e < tmp_end; ++e) {
          const node_index_t tmp_dst = edges[e].dst;

          if (level[tmp_dst] == MAX_LEVEL) { // Unmarked
            level[tmp_dst] = next_level;
            ++cnt;
          }
        }
      }
    }

    level_counts[next_level] = cnt;
    if (cnt == 0)
      break;
  }

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  bfs_kernel_time += (kernel_end.tv_sec - kernel_start.tv_sec) +
                     (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
