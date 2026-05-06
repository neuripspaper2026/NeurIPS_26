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

  // Initialize starting frontier
  level[starting_node] = 0;
  level_counts[0] = 1;

  // BFS over horizons (levels)
  loop_horizons: for( horizon=0; horizon<N_LEVELS; horizon++ ) {
    cnt = 0;

    // Add unmarked neighbors of the current horizon to the next horizon
    // Parallelize over nodes; each thread accumulates its own count and then
    // reduces into the shared cnt to avoid false sharing/contention.
#ifdef _OPENMP
    #pragma omp parallel default(none) shared(nodes, edges, level, horizon, level_counts) private(n, e) reduction(+:cnt)
#endif
    {
#ifdef _OPENMP
      #pragma omp for schedule(static)
#endif
      for( n=0; n<(node_index_t)N_NODES; n++ ) {
        if( level[n] == horizon ) {
          const edge_index_t tmp_begin = nodes[n].edge_begin;
          const edge_index_t tmp_end   = nodes[n].edge_end;

          for( e = tmp_begin; e < tmp_end; e++ ) {
            const node_index_t tmp_dst = edges[e].dst;
            // Load once to allow the compiler to optimize better
            level_t tmp_level = level[tmp_dst];

            if( tmp_level == MAX_LEVEL ) { // Unmarked
              // It is safe to allow benign races on writes of the same value
              // to the same location (typical BFS optimization).
              level[tmp_dst] = (level_t)(horizon + 1);
              ++cnt;
            }
          }
        }
      }
    }

    level_counts[horizon+1] = cnt;
    if( cnt == 0 )
      break;
  }

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  bfs_kernel_time += (kernel_end.tv_sec - kernel_start.tv_sec) +
                     (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
