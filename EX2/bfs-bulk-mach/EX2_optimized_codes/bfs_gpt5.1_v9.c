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

  // Initialize levels to MAX_LEVEL (unvisited) before BFS
  // This ensures correctness even if caller didn't pre-initialize.
  for (n = 0; n < N_NODES; ++n) {
    level[n] = MAX_LEVEL;
  }

  level[starting_node] = 0;
  level_counts[0] = 1;

  loop_horizons: for( horizon=0; horizon<N_LEVELS; horizon++ ) {
    cnt = 0;

#ifdef _OPENMP
    edge_index_t cnt_local = 0;

    // Parallelize over nodes in current frontier
    #pragma omp parallel for default(none) shared(nodes, edges, level, horizon) reduction(+:cnt_local)
    for( n=0; n<(node_index_t)N_NODES; n++ ) {
      if( level[n]==horizon ) {
        edge_index_t tmp_begin = nodes[n].edge_begin;
        edge_index_t tmp_end = nodes[n].edge_end;
        for( e=tmp_begin; e<tmp_end; e++ ) {
          node_index_t tmp_dst = edges[e].dst;
          // Check-and-set visited using atomic compare-and-swap
          if (level[tmp_dst] == MAX_LEVEL) {
            level_t expected = MAX_LEVEL;
            #pragma omp atomic compare
            if (level[tmp_dst] == expected) {
              level[tmp_dst] = (level_t)(horizon + 1);
              ++cnt_local;
            }
          }
        }
      }
    }
    cnt = cnt_local;
#else
    // Serial version
    loop_nodes: for( n=0; n<N_NODES; n++ ) {
      if( level[n]==horizon ) {
        edge_index_t tmp_begin = nodes[n].edge_begin;
        edge_index_t tmp_end = nodes[n].edge_end;
        loop_neighbors: for( e=tmp_begin; e<tmp_end; e++ ) {
          node_index_t tmp_dst = edges[e].dst;
          level_t tmp_level = level[tmp_dst];

          if( tmp_level == MAX_LEVEL ) { // Unmarked
            level[tmp_dst] = (level_t)(horizon+1);
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
