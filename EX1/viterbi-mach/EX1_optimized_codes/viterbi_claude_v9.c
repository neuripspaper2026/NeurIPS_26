#include <time.h>
#include "../viterbi.h"

static double viterbi_kernel_time_acc = 0.0;

void reset_viterbi_kernel_time(void) { viterbi_kernel_time_acc = 0.0; }
double get_viterbi_kernel_time(void) { return viterbi_kernel_time_acc; }

int viterbi( tok_t obs[N_OBS], prob_t init[N_STATES], prob_t transition[N_STATES*N_STATES], prob_t emission[N_STATES*N_TOKENS], state_t path[N_OBS] )
{
  prob_t llike[2][N_STATES];
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
    int curr_buf = t & 1;
    int prev_buf = 1 - curr_buf;
    tok_t obs_t = obs[t];
    
    L_curr_state: for( curr=0; curr<N_STATES; curr++ ) {
      prob_t emission_val = emission[curr*N_TOKENS+obs_t];
      state_t curr_offset = curr;
      
      prev = 0;
      min_p = llike[prev_buf][0] + transition[curr] + emission_val;
      
      L_prev_state: for( prev=1; prev<N_STATES; prev++ ) {
        p = llike[prev_buf][prev] + transition[prev*N_STATES+curr_offset] + emission_val;
        if( p<min_p ) {
          min_p = p;
        }
      }
      llike[curr_buf][curr] = min_p;
    }
  }

  // Identify end state
  int final_buf = (N_OBS-1) & 1;
  min_s = 0;
  min_p = llike[final_buf][0];
  L_end: for( s=1; s<N_STATES; s++ ) {
    p = llike[final_buf][s];
    if( p<min_p ) {
      min_p = p;
      min_s = s;
    }
  }
  path[N_OBS-1] = min_s;

  // Reconstruct llike for backtracking
  prob_t llike_full[N_OBS][N_STATES];
  
  obs0 = obs[0];
  for( s=0; s<N_STATES; s++ ) {
    llike_full[0][s] = init[s] + emission[s*N_TOKENS+obs0];
  }
  
  for( t=1; t<N_OBS; t++ ) {
    tok_t obs_t = obs[t];
    
    for( curr=0; curr<N_STATES; curr++ ) {
      prob_t emission_val = emission[curr*N_TOKENS+obs_t];
      
      min_p = llike_full[t-1][0] + transition[curr] + emission_val;
      
      for( prev=1; prev<N_STATES; prev++ ) {
        p = llike_full[t-1][prev] + transition[prev*N_STATES+curr] + emission_val;
        if( p<min_p ) {
          min_p = p;
        }
      }
      llike_full[t][curr] = min_p;
    }
  }

  // Backtrack to recover full path
  L_backtrack: for( t=N_OBS-2; t>=0; t-- ) {
    state_t next_state = path[t+1];
    min_s = 0;
    min_p = llike_full[t][0] + transition[next_state];
    L_state: for( s=1; s<N_STATES; s++ ) {
      p = llike_full[t][s] + transition[s*N_STATES+next_state];
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

