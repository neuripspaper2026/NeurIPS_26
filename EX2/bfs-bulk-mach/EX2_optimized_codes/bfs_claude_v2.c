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

  // Pre-allocate thread-local buffers for parallel reduction
#ifdef _OPENMP
  int max_threads = omp_get_max_threads();
  edge_index_t *thread_counts = (edge_index_t *)calloc(max_threads, sizeof(edge_index_t));
#endif

  loop_horizons: for( horizon=0; horizon<N_LEVELS; horizon++ ) {
    cnt = 0;
    
#ifdef _OPENMP
    // Reset thread-local counters
    for(int t = 0; t < max_threads; t++) {
      thread_counts[t] = 0;
    }
    
    // Parallel loop over nodes with dynamic scheduling for load balancing
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      edge_index_t local_cnt = 0;
      
      #pragma omp for schedule(dynamic, 16) nowait
      for( n=0; n<N_NODES; n++ ) {
        if( level[n]==horizon ) {
          edge_index_t tmp_begin = nodes[n].edge_begin;
          edge_index_t tmp_end = nodes[n].edge_end;
          
          // Prefetch next node data
          if(n + 1 < N_NODES) {
            __builtin_prefetch(&nodes[n+1], 0, 1);
            __builtin_prefetch(&level[n+1], 0, 1);
          }
          
          for( e=tmp_begin; e<tmp_end; e++ ) {
            node_index_t tmp_dst = edges[e].dst;
            
            // Prefetch destination level
            __builtin_prefetch(&level[tmp_dst], 1, 1);
            
            level_t tmp_level = level[tmp_dst];
            
            if( tmp_level == MAX_LEVEL ) {
              // Atomic update to avoid race conditions
              level_t expected = MAX_LEVEL;
              if(__atomic_compare_exchange_n(&level[tmp_dst], &expected, horizon+1, 
                                            0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
                ++local_cnt;
              }
            }
          }
        }
      }
      
      // Store thread-local count
      thread_counts[tid] = local_cnt;
    }
    
    // Reduce thread-local counts
    for(int t = 0; t < max_threads; t++) {
      cnt += thread_counts[t];
    }
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

#ifdef _OPENMP
  free(thread_counts);
#endif

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  bfs_kernel_time += (kernel_end.tv_sec - kernel_start.tv_sec) +
                     (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
