#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../bfs.h"

static double bfs_queue_kernel_time_acc = 0.0;

void reset_bfs_queue_kernel_time(void) { bfs_queue_kernel_time_acc = 0.0; }
double get_bfs_queue_kernel_time(void) { return bfs_queue_kernel_time_acc; }

/* Ring buffer queue macros:
 * Head (q_out) points to the current front element.
 * Tail (q_in) points to the slot where the next element will be enqueued.
 * Queue is empty when q_in == q_out.
 * Queue is full when advancing q_in would make it equal to q_out.
 */
#define Q_PUSH(node)               \
  do {                             \
    queue[q_in] = (node);          \
    q_in = (q_in + 1) & (N_NODES - 1); \
  } while (0)

#define Q_PEEK() (queue[q_out])

#define Q_POP()                    \
  do {                             \
    q_out = (q_out + 1) & (N_NODES - 1); \
  } while (0)

#define Q_EMPTY() (q_in == q_out)

void bfs(node_t nodes[N_NODES], edge_t edges[N_EDGES],
         node_index_t starting_node, level_t level[N_NODES],
         edge_index_t level_counts[N_LEVELS])
{
  node_index_t queue[N_NODES];
  node_index_t q_in, q_out;
  node_index_t dummy;
  node_index_t n;
  edge_index_t e;
  struct timespec kernel_start, kernel_end;

  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  /* Initialize queue as empty */
  q_in = 0;
  q_out = 0;

  /* Initialize starting node */
  level[starting_node] = 0;
  level_counts[0] = 1;
  Q_PUSH(starting_node);

  /* Main BFS loop: processes at most N_NODES dequeues */
  loop_queue:
  for (dummy = 0; dummy < N_NODES; dummy++) {
    if (Q_EMPTY())
      break;

    n = Q_PEEK();
    Q_POP();

    edge_index_t tmp_begin = nodes[n].edge_begin;
    edge_index_t tmp_end = nodes[n].edge_end;
    level_t base_level = level[n] + 1;

    /* Traverse adjacency list of node n */
    loop_neighbors:
    for (e = tmp_begin; e < tmp_end; e++) {
      node_index_t tmp_dst = edges[e].dst;

      /* Fast path: check first to avoid unnecessary reloads */
      if (level[tmp_dst] == MAX_LEVEL) {
        level[tmp_dst] = base_level;
#ifdef _OPENMP
        /* Level counts can be updated atomically to be thread-safe
         * if this region is later parallelized (e.g., via tasks).
         * In current serial execution, atomics are cheap no-ops.
         */
#pragma omp atomic
#endif
        ++level_counts[base_level];
        Q_PUSH(tmp_dst);
      }
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  bfs_queue_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                               (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
