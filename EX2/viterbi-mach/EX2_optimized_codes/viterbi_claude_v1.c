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
  struct timespec kernel_start, kernel_end;

  clock_gettime(CLOCK_MONOTONIC, &kernel_start);
 
  // Initialize with first observation and initial probabilities
  #pragma omp parallel for schedule(static)
  for( s=0; s<N_STATES; s++ ) {
    llike[0][s] = init[s] + emission[s*N_TOKENS+obs[0]];
  }

  // Iteratively compute the probabilities over time
  for( t=1; t<N_OBS; t++ ) {
    tok_t obs_t = obs[t];
    #pragma omp parallel for schedule(static) private(prev, min_p, p)
    for( curr=0; curr<N_STATES; curr++ ) {
      prob_t emission_curr_obs = emission[curr*N_TOKENS+obs_t];
      
      // Compute likelihood HMM is in current state and where it came from.
      min_p = llike[t-1][0] + transition[curr] + emission_curr_obs;
      
      #pragma omp simd reduction(min:min_p)
      for( prev=1; prev<N_STATES; prev++ ) {
        p = llike[t-1][prev] + transition[prev*N_STATES+curr] + emission_curr_obs;
        if( p < min_p ) {
          min_p = p;
        }
      }
      llike[t][curr] = min_p;
    }
  }

  // Identify end state
  min_s = 0;
  min_p = llike[N_OBS-1][0];
  for( s=1; s<N_STATES; s++ ) {
    p = llike[N_OBS-1][s];
    if( p<min_p ) {
      min_p = p;
      min_s = s;
    }
  }
  path[N_OBS-1] = min_s;

  // Backtrack to recover full path
  for( t=N_OBS-2; t>=0; t-- ) {
    state_t path_next = path[t+1];
    min_s = 0;
    min_p = llike[t][0] + transition[path_next];
    for( s=1; s<N_STATES; s++ ) {
      p = llike[t][s] + transition[s*N_STATES+path_next];
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
  return 0;
}

