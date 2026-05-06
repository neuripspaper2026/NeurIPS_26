#include <time.h>
#include "../bfs.h"

static double bfs_queue_kernel_time_acc = 0.0;

void reset_bfs_queue_kernel_time(void) { bfs_queue_kernel_time_acc = 0.0; }
double get_bfs_queue_kernel_time(void) { return bfs_queue_kernel_time_acc; }

#define Q_PUSH(node) do {                               \
  queue[q_in] = (node);                                 \
  q_in = (node_index_t)((q_in + 1u) & (N_NODES - 1u));  \
} while (0)

#define Q_PEEK() (queue[q_out])

#define Q_POP() do {                                    \
  q_out = (node_index_t)((q_out + 1u) & (N_NODES - 1u));\
} while (0)

#define Q_EMPTY() (q_in == q_out)

void bfs(node_t nodes[N_NODES], edge_t edges[N_EDGES],
         node_index_t starting_node, level_t level[N_NODES],
         edge_index_t level_counts[N_LEVELS])
{
  node_index_t queue[N_NODES];
  node_index_t q_in = 0;
  node_index_t q_out = 0;
  node_index_t n;
  edge_index_t e;
  struct timespec kernel_start, kernel_end;

  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  level[starting_node] = 0;
  level_counts[0] = 1;
  Q_PUSH(starting_node);

  while (!Q_EMPTY()) {
    n = Q_PEEK();
    Q_POP();
    edge_index_t tmp_begin = nodes[n].edge_begin;
    edge_index_t tmp_end = nodes[n].edge_end;

    for (e = tmp_begin; e < tmp_end; ++e) {
      node_index_t tmp_dst = edges[e].dst;
      level_t tmp_level = level[tmp_dst];

      if (tmp_level == MAX_LEVEL) {
        level_t new_level = (level_t)(level[n] + 1);
        level[tmp_dst] = new_level;
        ++level_counts[(int)new_level];
        Q_PUSH(tmp_dst);
      }
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  bfs_queue_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                               (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
