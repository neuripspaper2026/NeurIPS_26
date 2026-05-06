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

  // Prefetch nodes array into cache
  #ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  #endif
  for(n = 0; n < N_NODES; n++) {
    __builtin_prefetch(&nodes[n], 0, 3);
  }

  loop_horizons: for( horizon=0; horizon<N_LEVELS; horizon++ ) {
    cnt = 0;
    
    #ifdef _OPENMP
    edge_index_t local_cnt = 0;
    #pragma omp parallel private(n, e) reduction(+:local_cnt)
    {
      node_index_t local_n;
      edge_index_t local_e;
      edge_index_t tmp_begin, tmp_end;
      node_index_t tmp_dst;
      level_t tmp_level;
      
      #pragma omp for schedule(dynamic, 32) nowait
      for( local_n=0; local_n<N_NODES; local_n++ ) {
        if( level[local_n]==horizon ) {
          tmp_begin = nodes[local_n].edge_begin;
          tmp_end = nodes[local_n].edge_end;
          
          // Prefetch edge data
          if(tmp_end > tmp_begin) {
            __builtin_prefetch(&edges[tmp_begin], 0, 2);
          }
          
          for( local_e=tmp_begin; local_e<tmp_end; local_e++ ) {
            // Prefetch next edge
            if(local_e + 1 < tmp_end) {
              __builtin_prefetch(&edges[local_e + 1], 0, 2);
            }
            
            tmp_dst = edges[local_e].dst;
            
            // Prefetch destination level
            __builtin_prefetch(&level[tmp_dst], 1, 2);
            
            tmp_level = level[tmp_dst];

            if( tmp_level == MAX_LEVEL ) {
              level[tmp_dst] = horizon+1;
              local_cnt++;
            }
          }
        }
      }
    }
    cnt = local_cnt;
    #else
    // Serial version with optimizations
    loop_nodes: for( n=0; n<N_NODES; n++ ) {
      if( level[n]==horizon ) {
        edge_index_t tmp_begin = nodes[n].edge_begin;
        edge_index_t tmp_end = nodes[n].edge_end;
        
        // Prefetch edge data
        if(tmp_end > tmp_begin) {
          __builtin_prefetch(&edges[tmp_begin], 0, 3);
        }
        
        loop_neighbors: for( e=tmp_begin; e<tmp_end; e++ ) {
          // Prefetch next edge
          if(e + 1 < tmp_end) {
            __builtin_prefetch(&edges[e + 1], 0, 3);
          }
          
          node_index_t tmp_dst = edges[e].dst;
          
          // Prefetch destination level
          __builtin_prefetch(&level[tmp_dst], 1, 3);
          
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
