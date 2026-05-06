#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../viterbi.h"

static double viterbi_kernel_time_acc = 0.0;

void reset_viterbi_kernel_time(void) { viterbi_kernel_time_acc = 0.0; }
double get_viterbi_kernel_time(void) { return viterbi_kernel_time_acc; }

int viterbi( tok_t obs[N_OBS], prob_t init[N_STATES], prob_t transition[N_STATES*N_STATES], prob_t emission[N_STATES*N_TOKENS], state_t path[N_OBS] )
{
  prob_t llike[N_OBS][N_STATES];
  step_t t;
  state_t prev, curr;
  prob_t min_p, p;
  state_t min_s, s;
  int result = 0;
  // All probabilities are in -log space. (i.e.: P(x) => -log(P(x)) )
  struct timespec kernel_start, kernel_end;

  clock_gettime(CLOCK_MONOTONIC, &kernel_start);
 
  // Initialize with first observation and initial probabilities
  #pragma omp parallel for
  L_init: for( s=0; s<N_STATES; s++ ) {
    llike[0][s] = init[s] + emission[s*N_TOKENS+obs[0]];
  }

  // Iteratively compute the probabilities over time
  L_timestep: for( t=1; t<N_OBS; t++ ) {
    #pragma omp parallel for private(curr, prev, min_p, p)
    L_curr_state: for( curr=0; curr<N_STATES; curr++ ) {
      // Compute likelihood HMM is in current state and where it came from.
      prev = 0;
      min_p = llike[t-1][prev] +
              transition[prev*N_STATES+curr] +
              emission[curr*N_TOKENS+obs[t]];
      L_prev_state: for( prev=1; prev<N_STATES; prev++ ) {
        p = llike[t-1][prev] +
            transition[prev*N_STATES+curr] +
            emission[curr*N_TOKENS+obs[t]];
        if( p<min_p ) {
          min_p = p;
        }
      }
      llike[t][curr] = min_p;
    }
  }

  // Identify end state
  min_s = 0;
  min_p = llike[N_OBS-1][min_s];
  L_end: for( s=1; s<N_STATES; s++ ) {
    p = llike[N_OBS-1][s];
    if( p<min_p ) {
      min_p = p;
      min_s = s;
    }
  }
  path[N_OBS-1] = min_s;

  // Backtrack to recover full path
  L_backtrack: for( t=N_OBS-2; t>=0; t-- ) {
    min_s = 0;
    min_p = llike[t][min_s] + transition[min_s*N_STATES+path[t+1]];
    L_state: for( s=1; s<N_STATES; s++ ) {
      p = llike[t][s] + transition[s*N_STATES+path[t+1]];
      if( p<min_p ) {
        min_p = p;
        min_s = s;
      }
    }
    path[t] = min_s;
  }

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  viterbi_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                             (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
  return result;
}

