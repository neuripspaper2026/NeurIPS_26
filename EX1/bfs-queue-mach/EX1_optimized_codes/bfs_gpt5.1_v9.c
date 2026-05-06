#include <time.h>
#include "../bfs.h"

static double bfs_queue_kernel_time_acc = 0.0;

void reset_bfs_queue_kernel_time(void) { bfs_queue_kernel_time_acc = 0.0; }
double get_bfs_queue_kernel_time(void) { return bfs_queue_kernel_time_acc; }

#define Q_PUSH(node)         \
  do {                       \
    queue[q_in] = (node);    \
    q_in = (q_in + 1);       \
  } while (0)

#define Q_PEEK() (queue[q_out])

#define Q_POP()      \
  do {               \
    q_out = q_out+1; \
  } while (0)

#define Q_EMPTY() (q_in == q_out)

void bfs(node_t nodes[N_NODES], edge_t edges[N_EDGES],
         node_index_t starting_node, level_t level[N_NODES],
         edge_index_t level_counts[N_LEVELS])
{
  node_index_t queue[N_NODES + 1]; // +1 to allow simple index wrap
  node_index_t q_in, q_out;
  node_index_t dummy;
  node_index_t n;
  edge_index_t e;
  struct timespec kernel_start, kernel_end;

  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  // Use indices in [0, N_NODES]; wrap by manual checks
  q_in = 0;
  q_out = 0;

  // Initialize first level
  level[starting_node] = 0;
  level_counts[0] = 1;

  Q_PUSH(starting_node);

  for (dummy = 0; dummy < N_NODES; dummy++) { // acts as safety bound
    if (Q_EMPTY())
      break;

    n = Q_PEEK();
    Q_POP();

    if (q_out == N_NODES + 1)
      q_out = 0;

    edge_index_t tmp_begin = nodes[n].edge_begin;
    edge_index_t tmp_end = nodes[n].edge_end;

    for (e = tmp_begin; e < tmp_end; e++) {
      node_index_t tmp_dst = edges[e].dst;
      level_t cur_level = level[tmp_dst];

      if (cur_level == MAX_LEVEL) { // Unmarked
        level_t new_level = (level_t)(level[n] + 1);
        level[tmp_dst] = new_level;
        ++level_counts[new_level];

        Q_PUSH(tmp_dst);
        if (q_in == N_NODES + 1)
          q_in = 0;
      }
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  bfs_queue_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                               (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
