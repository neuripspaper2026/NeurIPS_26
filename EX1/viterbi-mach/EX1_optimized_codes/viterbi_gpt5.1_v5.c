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
  const int n_states = N_STATES;
  const int n_obs = N_OBS;
  const int n_tokens = N_TOKENS;
  // All probabilities are in -log space. (i.e.: P(x) => -log(P(x)) )
  struct timespec kernel_start, kernel_end;

  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  // Initialize with first observation and initial probabilities
  {
    const tok_t o0 = obs[0];
    const int base_em = o0;
    for( s = 0; s < n_states; s++ ) {
      llike[0][s] = init[s] + emission[s*n_tokens + base_em];
    }
  }

  // Iteratively compute the probabilities over time
  for( t = 1; t < n_obs; t++ ) {
    const tok_t ot = obs[t];
    const int emit_off = ot;
    for( curr = 0; curr < n_states; curr++ ) {
      const int trans_col = curr;
      const int emit_idx = curr*n_tokens + emit_off;
      const prob_t emit_val = emission[emit_idx];

      prev = 0;
      prob_t best = llike[t-1][0] +
                    transition[0*n_states + trans_col] +
                    emit_val;

      for( prev = 1; prev < n_states; prev++ ) {
        p = llike[t-1][prev] +
            transition[prev*n_states + trans_col] +
            emit_val;
        if( p < best ) {
          best = p;
        }
      }
      llike[t][curr] = best;
    }
  }

  // Identify end state
  {
    const int last_t = n_obs - 1;
    min_s = 0;
    min_p = llike[last_t][0];
    for( s = 1; s < n_states; s++ ) {
      p = llike[last_t][s];
      if( p < min_p ) {
        min_p = p;
        min_s = s;
      }
    }
    path[last_t] = min_s;
  }

  // Backtrack to recover full path
  for( t = n_obs - 2; t >= 0; t-- ) {
    const int next_state = path[t+1];
    const int trans_row0 = 0*n_states + next_state;
    min_s = 0;
    min_p = llike[t][0] + transition[trans_row0];
    for( s = 1; s < n_states; s++ ) {
      p = llike[t][s] + transition[s*n_states + next_state];
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

