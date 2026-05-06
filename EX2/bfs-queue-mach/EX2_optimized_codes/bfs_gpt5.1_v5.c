#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../bfs.h"

static double bfs_queue_kernel_time_acc = 0.0;

void reset_bfs_queue_kernel_time(void) { bfs_queue_kernel_time_acc = 0.0; }
double get_bfs_queue_kernel_time(void) { return bfs_queue_kernel_time_acc; }

/* Use a simple circular queue; keep macros small and side‑effect free where possible */
#define Q_PUSH(node)                     \
  do {                                   \
    queue[q_in] = (node);                \
    q_in = (q_in + 1) & (N_NODES - 1);   \
  } while (0)

#define Q_PEEK() (queue[q_out])

#define Q_POP()                          \
  do {                                   \
    q_out = (q_out + 1) & (N_NODES - 1); \
  } while (0)

/* For N_NODES being a power of two this is efficient and branch‑light */
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

  /* Initialize queue as empty (in == out) and push starting node */
  q_in = 0;
  q_out = 0;

  /* Initialize BFS levels and level_counts in a cache‑friendly way */
  /* Mark all nodes as unvisited */
  for (node_index_t i = 0; i < N_NODES; ++i) {
    level[i] = MAX_LEVEL;
  }

  /* Initialize level_counts */
  for (int i = 0; i < N_LEVELS; ++i) {
    level_counts[i] = 0;
  }

  level[starting_node] = 0;
  level_counts[0] = 1;
  Q_PUSH(starting_node);

  /* Standard BFS using a queue; body is kept simple to expose ILP */
  loop_queue:
  for (dummy = 0; dummy < N_NODES; ++dummy) {
    if (Q_EMPTY())
      break;

    n = Q_PEEK();
    Q_POP();

    const edge_index_t tmp_begin = nodes[n].edge_begin;
    const edge_index_t tmp_end   = nodes[n].edge_end;
    const level_t next_level     = (level_t)(level[n] + 1);

    /* Sequential neighbor expansion; inner loop is amenable to compiler vectorization */
    loop_neighbors:
    for (e = tmp_begin; e < tmp_end; ++e) {
      const node_index_t tmp_dst = edges[e].dst;

      /* Check first to avoid unnecessary writes */
      if (level[tmp_dst] == MAX_LEVEL) {
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
