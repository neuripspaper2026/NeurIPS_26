#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../bfs.h"

static double bfs_queue_kernel_time_acc = 0.0;

void reset_bfs_queue_kernel_time(void) { bfs_queue_kernel_time_acc = 0.0; }
double get_bfs_queue_kernel_time(void) { return bfs_queue_kernel_time_acc; }

void bfs(node_t nodes[N_NODES], edge_t edges[N_EDGES],
            node_index_t starting_node, level_t level[N_NODES],
            edge_index_t level_counts[N_LEVELS])
{
  node_index_t *current_frontier;
  node_index_t *next_frontier;
  node_index_t current_size, next_size;
  node_index_t n;
  edge_index_t e;
  level_t current_level;
  struct timespec kernel_start, kernel_end;

  current_frontier = (node_index_t*)malloc(N_NODES * sizeof(node_index_t));
  next_frontier = (node_index_t*)malloc(N_NODES * sizeof(node_index_t));

  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  current_size = 1;
  current_frontier[0] = starting_node;
  level[starting_node] = 0;
  level_counts[0] = 1;
  current_level = 0;

  while(current_size > 0 && current_level < N_LEVELS - 1) {
    next_size = 0;

#ifdef _OPENMP
    #pragma omp parallel
    {
      node_index_t *local_next = (node_index_t*)malloc(N_NODES * sizeof(node_index_t));
      node_index_t local_next_size = 0;

      #pragma omp for schedule(dynamic, 64) nowait
      for(node_index_t i = 0; i < current_size; i++) {
        n = current_frontier[i];
        edge_index_t tmp_begin = nodes[n].edge_begin;
        edge_index_t tmp_end = nodes[n].edge_end;
        
        for(e = tmp_begin; e < tmp_end; e++) {
          node_index_t tmp_dst = edges[e].dst;
          level_t old_level = __atomic_load_n(&level[tmp_dst], __ATOMIC_ACQUIRE);
          
          if(old_level == MAX_LEVEL) {
            level_t expected = MAX_LEVEL;
            if(__atomic_compare_exchange_n(&level[tmp_dst], &expected, current_level + 1,
                                          0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
              local_next[local_next_size++] = tmp_dst;
            }
          }
        }
      }

      node_index_t offset;
      #pragma omp atomic capture
      {
        offset = next_size;
        next_size += local_next_size;
      }

      for(node_index_t i = 0; i < local_next_size; i++) {
        next_frontier[offset + i] = local_next[i];
      }

      free(local_next);
    }
#else
    for(node_index_t i = 0; i < current_size; i++) {
      n = current_frontier[i];
      edge_index_t tmp_begin = nodes[n].edge_begin;
      edge_index_t tmp_end = nodes[n].edge_end;
      
      for(e = tmp_begin; e < tmp_end; e++) {
        node_index_t tmp_dst = edges[e].dst;
        
        if(level[tmp_dst] == MAX_LEVEL) {
          level[tmp_dst] = current_level + 1;
          next_frontier[next_size++] = tmp_dst;
        }
      }
    }
#endif

    current_level++;
    if(next_size > 0) {
      level_counts[current_level] = next_size;
    }

    node_index_t *temp = current_frontier;
    current_frontier = next_frontier;
    next_frontier = temp;
    current_size = next_size;
  }

  free(current_frontier);
  free(next_frontier);

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  bfs_queue_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                               (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
