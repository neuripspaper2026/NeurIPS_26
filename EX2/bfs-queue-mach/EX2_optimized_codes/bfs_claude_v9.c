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
  node_index_t current_queue[N_NODES];
  node_index_t next_queue[N_NODES];
  node_index_t current_size, next_size;
  level_t current_level;
  struct timespec kernel_start, kernel_end;

  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  current_queue[0] = starting_node;
  current_size = 1;
  current_level = 0;
  level[starting_node] = 0;
  level_counts[0] = 1;

  while(current_size > 0 && current_level < N_LEVELS - 1) {
    next_size = 0;
    
#ifdef _OPENMP
    #pragma omp parallel
    {
      node_index_t local_next_queue[N_NODES];
      node_index_t local_next_size = 0;
      
      #pragma omp for schedule(dynamic, 16) nowait
      for(node_index_t i = 0; i < current_size; i++) {
        node_index_t n = current_queue[i];
        edge_index_t tmp_begin = nodes[n].edge_begin;
        edge_index_t tmp_end = nodes[n].edge_end;
        
        for(edge_index_t e = tmp_begin; e < tmp_end; e++) {
          node_index_t tmp_dst = edges[e].dst;
          
          if(level[tmp_dst] == MAX_LEVEL) {
            level_t old_level = __sync_val_compare_and_swap(&level[tmp_dst], MAX_LEVEL, current_level + 1);
            if(old_level == MAX_LEVEL) {
              local_next_queue[local_next_size++] = tmp_dst;
            }
          }
        }
      }
      
      #pragma omp critical
      {
        for(node_index_t i = 0; i < local_next_size; i++) {
          next_queue[next_size++] = local_next_queue[i];
        }
      }
    }
#else
    for(node_index_t i = 0; i < current_size; i++) {
      node_index_t n = current_queue[i];
      edge_index_t tmp_begin = nodes[n].edge_begin;
      edge_index_t tmp_end = nodes[n].edge_end;
      
      for(edge_index_t e = tmp_begin; e < tmp_end; e++) {
        node_index_t tmp_dst = edges[e].dst;
        
        if(level[tmp_dst] == MAX_LEVEL) {
          level[tmp_dst] = current_level + 1;
          next_queue[next_size++] = tmp_dst;
        }
      }
    }
#endif
    
    level_counts[current_level + 1] = next_size;
    
    node_index_t *temp = current_queue;
    current_queue = next_queue;
    next_queue = temp;
    current_size = next_size;
    current_level++;
  }

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  bfs_queue_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                               (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
