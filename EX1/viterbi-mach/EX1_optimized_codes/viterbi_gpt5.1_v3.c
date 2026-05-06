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

  // Precompute base pointers to reduce indexing cost
  prob_t *restrict llike0 = &llike[0][0];
  const prob_t *restrict trans = transition;
  const prob_t *restrict emit  = emission;
  const tok_t *restrict obs_r  = obs;
  const prob_t *restrict init_r = init;

  // Initialize with first observation and initial probabilities
  {
    const tok_t o0 = obs_r[0];
    const size_t emit_row_stride = N_TOKENS;
    for( s = 0; s < N_STATES; s++ ) {
      llike0[s] = init_r[s] + emit[(size_t)s*emit_row_stride + o0];
    }
  }

  // Iteratively compute the probabilities over time
  for( t = 1; t < N_OBS; t++ ) {
    const tok_t ot = obs_r[t];
    const size_t emit_row_stride = N_TOKENS;
    const size_t trans_row_stride = N_STATES;
    prob_t *restrict llike_prev = &llike[t-1][0];
    prob_t *restrict llike_curr = &llike[t][0];

    for( curr = 0; curr < N_STATES; curr++ ) {
      const prob_t emit_val = emit[(size_t)curr*emit_row_stride + ot];

      // prev = 0 specialization
      prob_t const *restrict trans_col_base = &trans[curr];
      prev = 0;
      min_p = llike_prev[0] + trans_col_base[0*trans_row_stride] + emit_val;

      // loop over remaining prev states
      for( prev = 1; prev < N_STATES; prev++ ) {
        p = llike_prev[prev] + trans_col_base[(size_t)prev*trans_row_stride] + emit_val;
        if( p < min_p ) {
          min_p = p;
        }
      }
      llike_curr[curr] = min_p;
    }
  }

  // Identify end state
  {
    prob_t *restrict llike_last = &llike[N_OBS-1][0];
    min_s = 0;
    min_p = llike_last[0];
    for( s = 1; s < N_STATES; s++ ) {
      p = llike_last[s];
      if( p < min_p ) {
        min_p = p;
        min_s = s;
      }
    }
    path[N_OBS-1] = min_s;
  }

  // Backtrack to recover full path
  for( t = N_OBS-2; t >= 0; t-- ) {
    prob_t *restrict llike_t = &llike[t][0];
    const state_t next_state = path[t+1];
    const size_t trans_row_stride = N_STATES;

    min_s = 0;
    min_p = llike_t[0] + trans[(size_t)0*trans_row_stride + next_state];
    for( s = 1; s < N_STATES; s++ ) {
      p = llike_t[s] + trans[(size_t)s*trans_row_stride + next_state];
      if( p < min_p ) {
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

