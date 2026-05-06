#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../bfs.h"

static double bfs_kernel_time = 0.0;

void reset_bfs_kernel_time(void) { bfs_kernel_time = 0.0; }
double get_bfs_kernel_time(void) { return bfs_kernel_time; }

void bfs(node_t nodes[N_NODES], edge_t edges[N_EDGES],
            node_index_t starting_node, level_t level[N_NODES],
            edge_index_t level_counts[N_LEVELS])
{
  node_index_t n;
  edge_index_t e;
  level_t horizon;
  edge_index_t cnt;
  struct timespec kernel_start, kernel_end;

  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  level[starting_node] = 0;
  level_counts[0] = 1;

  // Prefetch nodes array to L1/L2 cache
  #ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  #endif
  for(node_index_t i = 0; i < N_NODES; i++) {
    __builtin_prefetch(&nodes[i], 0, 3);
  }

  loop_horizons: for( horizon=0; horizon<N_LEVELS; horizon++ ) {
    cnt = 0;
    
    #ifdef _OPENMP
    edge_index_t local_cnt = 0;
    
    // Parallel region with dynamic scheduling for load balancing
    #pragma omp parallel private(n, e) reduction(+:local_cnt)
    {
      edge_index_t thread_cnt = 0;
      
      #pragma omp for schedule(dynamic, 16) nowait
      for( n=0; n<N_NODES; n++ ) {
        if( level[n]==horizon ) {
          edge_index_t tmp_begin = nodes[n].edge_begin;
          edge_index_t tmp_end = nodes[n].edge_end;
          
          // Prefetch next node's data
          if(n + 1 < N_NODES && level[n+1] == horizon) {
            __builtin_prefetch(&nodes[n+1], 0, 2);
          }
          
          for( e=tmp_begin; e<tmp_end; e++ ) {
            node_index_t tmp_dst = edges[e].dst;
            
            // Prefetch destination level
            __builtin_prefetch(&level[tmp_dst], 0, 1);
            
            level_t tmp_level = level[tmp_dst];

            if( tmp_level == MAX_LEVEL ) {
              level[tmp_dst] = horizon+1;
              thread_cnt++;
            }
          }
        }
      }
      
      local_cnt = thread_cnt;
    }
    
    cnt = local_cnt;
    
    #else
    // Serial version
    for( n=0; n<N_NODES; n++ ) {
      if( level[n]==horizon ) {
        edge_index_t tmp_begin = nodes[n].edge_begin;
        edge_index_t tmp_end = nodes[n].edge_end;
        
        for( e=tmp_begin; e<tmp_end; e++ ) {
          node_index_t tmp_dst = edges[e].dst;
          level_t tmp_level = level[tmp_dst];

          if( tmp_level == MAX_LEVEL ) {
            level[tmp_dst] = horizon+1;
            ++cnt;
          }
        }
      }
    }
    #endif
    
    if( (level_counts[horizon+1]=cnt)==0 )
      break;
  }

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  bfs_kernel_time += (kernel_end.tv_sec - kernel_start.tv_sec) +
                     (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
