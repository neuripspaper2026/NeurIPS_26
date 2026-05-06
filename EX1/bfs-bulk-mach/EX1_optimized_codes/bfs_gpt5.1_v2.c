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

  /* Initialize all nodes as unvisited (MAX_LEVEL) and level_counts to 0 */
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

  /* Track the frontier size for each horizon to skip scanning all nodes */
  loop_horizons:
  for (horizon = 0; horizon < N_LEVELS - 1; ++horizon) {
    cnt = 0;

    /* If no nodes at this horizon, we are done */
    if (level_counts[horizon] == 0)
      break;

    loop_nodes:
    for (n = 0; n < N_NODES; n++) {
      if (level[n] == horizon) {
        edge_index_t tmp_begin = nodes[n].edge_begin;
        edge_index_t tmp_end = nodes[n].edge_end;
        loop_neighbors:
        for (e = tmp_begin; e < tmp_end; e++) {
          node_index_t tmp_dst = edges[e].dst;
          if (level[tmp_dst] == MAX_LEVEL) { /* Unmarked */
            level[tmp_dst] = (level_t)(horizon + 1);
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
