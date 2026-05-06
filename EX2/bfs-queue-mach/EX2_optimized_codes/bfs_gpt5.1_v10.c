#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../bfs.h"

static double bfs_queue_kernel_time_acc = 0.0;

void reset_bfs_queue_kernel_time(void) { bfs_queue_kernel_time_acc = 0.0; }
double get_bfs_queue_kernel_time(void) { return bfs_queue_kernel_time_acc; }

#define Q_PUSH(node)                           \
  do {                                         \
    queue[q_in == 0 ? N_NODES - 1 : q_in - 1] = (node); \
    q_in = (q_in + 1) & (N_NODES - 1);         \
  } while (0)

#define Q_PEEK() (queue[q_out])

#define Q_POP()                                \
  do {                                         \
    q_out = (q_out + 1) & (N_NODES - 1);       \
  } while (0)

#define Q_EMPTY() (q_in > q_out ? (q_in == q_out + 1) : ((q_in == 0) && (q_out == N_NODES - 1)))

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

  q_in = 1;
  q_out = 0;

  /* Initialize levels and level_counts efficiently */
#ifdef _OPENMP
  {
    int nthreads = 1;
#pragma omp parallel
    {
#pragma omp single
      nthreads = omp_get_num_threads();
    }

    /* Parallel initialization of level array */
#pragma omp parallel for schedule(static)
    for (node_index_t i = 0; i < N_NODES; ++i) {
      level[i] = MAX_LEVEL;
    }

    /* Parallel zeroing of level_counts except level 0, which we set below */
#pragma omp parallel for schedule(static)
    for (int i = 1; i < N_LEVELS; ++i) {
      level_counts[i] = 0;
    }
  }
#else
  for (node_index_t i = 0; i < N_NODES; ++i) {
    level[i] = MAX_LEVEL;
  }
  for (int i = 1; i < N_LEVELS; ++i) {
    level_counts[i] = 0;
  }
#endif

  level[starting_node] = 0;
  level_counts[0] = 1;
  Q_PUSH(starting_node);

  while (!Q_EMPTY()) {
    n = Q_PEEK();
    Q_POP();

    level_t current_level = level[n];
    edge_index_t tmp_begin = nodes[n].edge_begin;
    edge_index_t tmp_end = nodes[n].edge_end;

#ifdef _OPENMP
    /* Parallelize over adjacency list of node n when sufficiently large */
    edge_index_t deg = tmp_end - tmp_begin;
    if (deg > 32) {
#pragma omp parallel for schedule(static)
      for (edge_index_t idx = 0; idx < deg; ++idx) {
        edge_index_t e_local = tmp_begin + idx;
        node_index_t tmp_dst = edges[e_local].dst;

        if (__atomic_load_n(&level[tmp_dst], __ATOMIC_RELAXED) == MAX_LEVEL) {
          level_t new_level = current_level + 1;

          /* Atomically mark and enqueue if still unmarked */
          level_t expected = MAX_LEVEL;
          if (__atomic_compare_exchange_n(&level[tmp_dst], &expected, new_level,
                                          0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
#pragma omp atomic
            level_counts[new_level]++;

            /* Queue is shared; protect push with critical to preserve semantics */
#pragma omp critical(bfs_queue_push)
            {
              Q_PUSH(tmp_dst);
            }
          }
        }
      }
    } else
#endif
    {
      for (e = tmp_begin; e < tmp_end; ++e) {
        node_index_t tmp_dst = edges[e].dst;
        level_t tmp_level = level[tmp_dst];

        if (tmp_level == MAX_LEVEL) {
          level_t new_level = current_level + 1;
          level[tmp_dst] = new_level;
          ++level_counts[new_level];
          Q_PUSH(tmp_dst);
        }
      }
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  bfs_queue_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                               (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
