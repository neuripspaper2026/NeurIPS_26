#include <time.h>
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
  tok_t obs0 = obs[0];
  L_init: for( s=0; s<N_STATES; s++ ) {
    llike[0][s] = init[s] + emission[s*N_TOKENS+obs0];
  }

  // Iteratively compute the probabilities over time
  L_timestep: for( t=1; t<N_OBS; t++ ) {
    tok_t obs_t = obs[t];
    prob_t emission_offset = obs_t;
    
    L_curr_state: for( curr=0; curr<N_STATES; curr++ ) {
      prob_t emission_curr = emission[curr*N_TOKENS+emission_offset];
      prob_t *llike_prev = llike[t-1];
      const prob_t *trans_curr = &transition[curr];
      
      min_p = llike_prev[0] + trans_curr[0] + emission_curr;
      
      L_prev_state: for( prev=1; prev<N_STATES; prev++ ) {
        p = llike_prev[prev] + trans_curr[prev*N_STATES] + emission_curr;
        if( p<min_p ) {
          min_p = p;
        }
      }
      llike[t][curr] = min_p;
    }
  }

  // Identify end state
  prob_t *llike_end = llike[N_OBS-1];
  min_s = 0;
  min_p = llike_end[0];
  L_end: for( s=1; s<N_STATES; s++ ) {
    p = llike_end[s];
    if( p<min_p ) {
      min_p = p;
      min_s = s;
    }
  }
  path[N_OBS-1] = min_s;

  // Backtrack to recover full path
  L_backtrack: for( t=N_OBS-2; t>=0; t-- ) {
    state_t path_next = path[t+1];
    prob_t *llike_t = llike[t];
    
    min_s = 0;
    min_p = llike_t[0] + transition[path_next];
    
    L_state: for( s=1; s<N_STATES; s++ ) {
      p = llike_t[s] + transition[s*N_STATES+path_next];
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

