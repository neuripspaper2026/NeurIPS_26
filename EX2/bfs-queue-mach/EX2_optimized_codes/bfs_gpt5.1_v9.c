#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../bfs.h"

static double bfs_queue_kernel_time_acc = 0.0;

void reset_bfs_queue_kernel_time(void) { bfs_queue_kernel_time_acc = 0.0; }
double get_bfs_queue_kernel_time(void) { return bfs_queue_kernel_time_acc; }

#define Q_PUSH(node)            \
  do {                          \
    queue[q_in] = (node);       \
    q_in = (q_in + 1) & (N_NODES - 1); \
  } while (0)

#define Q_PEEK()   (queue[q_out])
#define Q_POP()    do { q_out = (q_out + 1) & (N_NODES - 1); } while (0)
#define Q_EMPTY()  (q_in == q_out)

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

  /* Initialize queue as a power-of-two ring buffer */
  q_in  = 0;
  q_out = 0;

  /* Initialize level counts to zero except starting node level; this
     is kept serial to avoid parallel reduction overhead for very
     small N_NODES and N_LEVELS. */
  for (dummy = 0; dummy < N_LEVELS; ++dummy) {
    level_counts[dummy] = 0;
  }

#ifdef _OPENMP
  /* Parallel initialization of levels to MAX_LEVEL */
  #pragma omp parallel for schedule(static)
  for (dummy = 0; dummy < (node_index_t)N_NODES; ++dummy) {
    level[dummy] = MAX_LEVEL;
  }
#else
  for (dummy = 0; dummy < (node_index_t)N_NODES; ++dummy) {
    level[dummy] = MAX_LEVEL;
  }
#endif

  level[starting_node] = 0;
  level_counts[0] = 1;
  Q_PUSH(starting_node);

  /* Main BFS loop over queue.
     The queue itself is processed serially to avoid heavy
     synchronization, but inner neighbor scan benefits from serial
     optimizations such as loop hoisting and reduced memory traffic.
   */
  for (dummy = 0; dummy < (node_index_t)N_NODES; dummy++) {
    if (Q_EMPTY())
      break;

    n = Q_PEEK();
    Q_POP();

    const edge_index_t tmp_begin = nodes[n].edge_begin;
    const edge_index_t tmp_end   = nodes[n].edge_end;
    const level_t parent_level   = level[n];
    const level_t next_level     = parent_level + 1;

    /* Neighbor traversal kept serial to preserve BFS order;
       uses local temporaries to help the compiler optimize. */
    for (e = tmp_begin; e < tmp_end; e++) {
      const node_index_t tmp_dst = edges[e].dst;
      const level_t cur_level = level[tmp_dst];

      if (cur_level == MAX_LEVEL) {
        level[tmp_dst] = next_level;
        ++level_counts[next_level];
        Q_PUSH(tmp_dst);
      }
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  bfs_queue_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                               (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
