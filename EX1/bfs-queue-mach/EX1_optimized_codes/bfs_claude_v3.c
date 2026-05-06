#include <time.h>
#include "../bfs.h"

static double bfs_queue_kernel_time_acc = 0.0;

void reset_bfs_queue_kernel_time(void) { bfs_queue_kernel_time_acc = 0.0; }
double get_bfs_queue_kernel_time(void) { return bfs_queue_kernel_time_acc; }

void bfs(node_t nodes[N_NODES], edge_t edges[N_EDGES],
            node_index_t starting_node, level_t level[N_NODES],
            edge_index_t level_counts[N_LEVELS])
{
  node_index_t queue[N_NODES];
  node_index_t q_in, q_out;
  node_index_t n;
  edge_index_t e;
  struct timespec kernel_start, kernel_end;

  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  q_in = 1;
  q_out = 0;
  level[starting_node] = 0;
  level_counts[0] = 1;
  queue[0] = starting_node;

  for( node_index_t dummy=0; dummy<N_NODES; dummy++ ) {
    if( q_in == q_out )
      break;
    
    n = queue[q_out];
    q_out = (q_out + 1) & (N_NODES - 1);
    
    edge_index_t tmp_begin = nodes[n].edge_begin;
    edge_index_t tmp_end = nodes[n].edge_end;
    level_t n_level = level[n];
    level_t next_level = n_level + 1;
    
    for( e=tmp_begin; e<tmp_end; e++ ) {
      node_index_t tmp_dst = edges[e].dst;
      level_t tmp_level = level[tmp_dst];

      if( tmp_level == MAX_LEVEL ) {
        level[tmp_dst] = next_level;
        ++level_counts[next_level];
        queue[q_in] = tmp_dst;
        q_in = (q_in + 1) & (N_NODES - 1);
      }
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  bfs_queue_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                               (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
