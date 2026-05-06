#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../bfs.h"

static double bfs_queue_kernel_time_acc = 0.0;

void reset_bfs_queue_kernel_time(void) { bfs_queue_kernel_time_acc = 0.0; }
double get_bfs_queue_kernel_time(void) { return bfs_queue_kernel_time_acc; }

#define Q_PUSH(node) do {                                      \
  node_index_t _qin = q_in;                                    \
  node_index_t _pos = (_qin == 0 ? (node_index_t)(N_NODES - 1) \
                                 : (node_index_t)(_qin - 1));  \
  queue[_pos] = (node);                                        \
  q_in = (node_index_t)(_qin + 1);                             \
  if (q_in == (node_index_t)N_NODES) q_in = 0;                 \
} while(0)

#define Q_PEEK() (queue[q_out])

#define Q_POP() do {                    \
  q_out = (node_index_t)(q_out + 1);    \
  if (q_out == (node_index_t)N_NODES)   \
    q_out = 0;                          \
} while(0)

#define Q_EMPTY() (q_in>q_out ? (q_in==(node_index_t)(q_out+1)) : ((q_in==(node_index_t)0) && (q_out==(node_index_t)(N_NODES-1))))

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

  /* Initialize queue and levels */
  q_in = 1;
  q_out = 0;

  /* Initialize all node levels to MAX_LEVEL to mark them unvisited */
  #ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  #endif
  for (node_index_t i = 0; i < (node_index_t)N_NODES; ++i) {
    level[i] = MAX_LEVEL;
  }

  /* Initialize level_counts to 0 */
  #ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  #endif
  for (int i = 0; i < N_LEVELS; ++i) {
    level_counts[i] = 0;
  }

  level[starting_node] = 0;
  level_counts[0] = 1;
  Q_PUSH(starting_node);

  /* Main BFS queue loop */
  loop_queue:
  for (dummy = 0; dummy < (node_index_t)N_NODES; dummy++ ) {
    if (Q_EMPTY())
      break;

    n = Q_PEEK();
    Q_POP();

    const level_t curr_level = level[n];
    const edge_index_t tmp_begin = nodes[n].edge_begin;
    const edge_index_t tmp_end   = nodes[n].edge_end;

    loop_neighbors:
    for (e = tmp_begin; e < tmp_end; ++e) {
      const node_index_t tmp_dst = edges[e].dst;
      level_t dst_level = level[tmp_dst];

      if (dst_level == MAX_LEVEL) { /* Unmarked */
        level_t new_level = (level_t)(curr_level + 1);

        /* Atomic compare-and-swap to avoid races in parallel settings */
        #ifdef _OPENMP
        #pragma omp atomic read
        dst_level = level[tmp_dst];
        if (dst_level == MAX_LEVEL) {
          #pragma omp atomic write
          level[tmp_dst] = new_level;
          #pragma omp atomic
          level_counts[new_level]++;
          Q_PUSH(tmp_dst);
        }
        #else
        level[tmp_dst] = new_level;
        ++level_counts[new_level];
        Q_PUSH(tmp_dst);
        #endif
      }
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  bfs_queue_kernel_time_acc +=
      (kernel_end.tv_sec - kernel_start.tv_sec) +
      (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
