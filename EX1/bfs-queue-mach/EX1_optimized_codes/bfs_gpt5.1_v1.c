#include <time.h>
#include "../bfs.h"

static double bfs_queue_kernel_time_acc = 0.0;

void reset_bfs_queue_kernel_time(void) { bfs_queue_kernel_time_acc = 0.0; }
double get_bfs_queue_kernel_time(void) { return bfs_queue_kernel_time_acc; }

#define Q_PUSH(node)           \
  do {                         \
    queue[tail] = (node);      \
    tail = (tail + 1) & MASK;  \
  } while (0)

#define Q_PEEK()   (queue[head])
#define Q_POP()    do { head = (head + 1) & MASK; } while (0)
#define Q_EMPTY()  (head == tail)

void bfs(node_t nodes[N_NODES], edge_t edges[N_EDGES],
         node_index_t starting_node, level_t level[N_NODES],
         edge_index_t level_counts[N_LEVELS])
{
  node_index_t queue[N_NODES];
  node_index_t head, tail;
  node_index_t n;
  edge_index_t e;
  struct timespec kernel_start, kernel_end;

  /* Precompute mask for power-of-two queue size */
  const node_index_t MASK = (node_index_t)(N_NODES - 1);

  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  head = 0;
  tail = 0;

  /* Initialize start node */
  level[starting_node] = 0;
  level_counts[0] = 1;
  Q_PUSH(starting_node);

  /* BFS queue processing */
  while (!Q_EMPTY()) {
    n = Q_PEEK();
    Q_POP();

    const edge_index_t tmp_begin = nodes[n].edge_begin;
    const edge_index_t tmp_end   = nodes[n].edge_end;
    const level_t base_level     = level[n] + 1;

    for (e = tmp_begin; e < tmp_end; ++e) {
      const node_index_t tmp_dst = edges[e].dst;

      if (level[tmp_dst] == MAX_LEVEL) { /* Unmarked */
        level[tmp_dst] = base_level;
        ++level_counts[base_level];
        Q_PUSH(tmp_dst);
      }
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  bfs_queue_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                               (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
