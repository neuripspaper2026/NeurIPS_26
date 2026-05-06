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

  /* Initialization of starting node and level counts */
  level[starting_node] = 0;
  /* Ensure other nodes are considered unvisited if not already; rely on caller for full init */
  level_counts[0] = 1;

  loop_horizons: for( horizon=0; horizon<N_LEVELS; horizon++ ) {
    cnt = 0;

#ifdef _OPENMP
    /* Parallel reduction over nodes; each thread processes a subset of nodes. */
    #pragma omp parallel for schedule(dynamic, 8) reduction(+:cnt)
#endif
    loop_nodes: for( n=0; n<(node_index_t)N_NODES; n++ ) {
      /* Fast path: skip nodes not in current frontier */
      if( level[n] == horizon ) {
        edge_index_t tmp_begin = nodes[n].edge_begin;
        edge_index_t tmp_end   = nodes[n].edge_end;

        /* Iterate neighbors of node n */
        loop_neighbors: for( e=tmp_begin; e<tmp_end; e++ ) {
          node_index_t tmp_dst = edges[e].dst;

          /* Read once to a local variable to help the compiler */
          level_t tmp_level = level[tmp_dst];

          /* Unmarked node: attempt to set its level */
          if( tmp_level == MAX_LEVEL ) {
#ifdef _OPENMP
            /* Use OpenMP atomic compare-and-swap style update to avoid races.
               Only the first thread that successfully updates the level counts it. */
            level_t expected = MAX_LEVEL;
            #pragma omp atomic compare
            if (level[tmp_dst] == expected) {
              level[tmp_dst] = (level_t)(horizon + 1);
              ++cnt;
            }
#else
            level[tmp_dst] = (level_t)(horizon + 1);
            ++cnt;
#endif
          }
        }
      }
    }

    /* Record number of nodes discovered at next level and terminate if none. */
    if( (level_counts[horizon+1] = cnt) == 0 )
      break;
  }

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  bfs_kernel_time += (kernel_end.tv_sec - kernel_start.tv_sec) +
                     (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
