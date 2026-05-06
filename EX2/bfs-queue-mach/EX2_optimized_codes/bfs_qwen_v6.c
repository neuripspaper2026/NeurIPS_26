#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../bfs.h"

static double bfs_queue_kernel_time_acc = 0.0;

void reset_bfs_queue_kernel_time(void) { bfs_queue_kernel_time_acc = 0.0; }
double get_bfs_queue_kernel_time(void) { return bfs_queue_kernel_time_acc; }

#define Q_PUSH(node) { queue[q_in==0?N_NODES-1:q_in-1]=node; q_in=(q_in+1)%N_NODES; }
#define Q_PEEK() (queue[q_out])
#define Q_POP() { q_out = (q_out+1)%N_NODES; }
#define Q_EMPTY() (q_in>q_out ? q_in==q_out+1 : (q_in==0)&&(q_out==N_NODES-1))

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

  // Initialize level array
  #pragma omp parallel for
  for (int i = 0; i < N_NODES; i++) {
    level[i] = MAX_LEVEL;
  }

  q_in = 1;
  q_out = 0;
  level[starting_node] = 0;
  level_counts[0] = 1;
  Q_PUSH(starting_node);

  loop_queue: for( dummy=0; dummy<N_NODES; dummy++ ) { // Typically while(not_empty(queue)){
    if( Q_EMPTY() )
      break;
    n = Q_PEEK();
    Q_POP();
    edge_index_t tmp_begin = nodes[n].edge_begin;
    edge_index_t tmp_end = nodes[n].edge_end;
    
    #ifdef _OPENMP
    #pragma omp parallel for shared(nodes, edges, level, level_counts, queue, q_in) private(e) schedule(dynamic, 64)
    #endif
    loop_neighbors: for( e=tmp_begin; e<tmp_end; e++ ) {
      node_index_t tmp_dst = edges[e].dst;
      level_t tmp_level = level[tmp_dst];

      if( tmp_level ==MAX_LEVEL ) { // Unmarked
        level_t new_level = level[n]+1;
        #ifdef _OPENMP
        if (__sync_bool_compare_and_swap(&level[tmp_dst], MAX_LEVEL, new_level)) {
          __sync_fetch_and_add(&level_counts[new_level], 1);
          #pragma omp critical
          {
            Q_PUSH(tmp_dst);
          }
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
  bfs_queue_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                               (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
