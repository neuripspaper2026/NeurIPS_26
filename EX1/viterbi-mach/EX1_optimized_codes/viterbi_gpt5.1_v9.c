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

  const int n_states = N_STATES;
  const int n_obs    = N_OBS;
  const int n_tokens = N_TOKENS;

  // Initialize with first observation and initial probabilities
  {
    const tok_t o0 = obs[0];
    const prob_t *restrict em_base = &emission[o0];
    prob_t *restrict llike0 = llike[0];
    for( s = 0; s < n_states; s++ ) {
      llike0[s] = init[s] + em_base[s * n_tokens];
    }
  }

  // Iteratively compute the probabilities over time
  for( t = 1; t < n_obs; t++ ) {
    const tok_t ot = obs[t];
    const prob_t *restrict em_col = &emission[ot];
    prob_t *restrict llike_prev = llike[t-1];
    prob_t *restrict llike_curr = llike[t];

    for( curr = 0; curr < n_states; curr++ ) {
      const prob_t emit_curr = em_col[curr * n_tokens];
      const prob_t *restrict trans_col = &transition[curr];
      prev = 0;
      const prob_t base0 = llike_prev[0] + trans_col[0 * n_states];
      min_p = base0 + emit_curr;

      for( prev = 1; prev < n_states; prev++ ) {
        p = llike_prev[prev] + trans_col[prev * n_states] + emit_curr;
        if( p < min_p ) {
          min_p = p;
        }
      }
      llike_curr[curr] = min_p;
    }
  }

  // Identify end state
  {
    const prob_t *restrict llike_last = llike[n_obs-1];
    min_s = 0;
    min_p = llike_last[min_s];
    for( s = 1; s < n_states; s++ ) {
      p = llike_last[s];
      if( p < min_p ) {
        min_p = p;
        min_s = s;
      }
    }
    path[n_obs-1] = min_s;
  }

  // Backtrack to recover full path
  for( t = n_obs-2; t >= 0; t-- ) {
    prob_t *restrict llike_t = llike[t];
    const state_t next_state = path[t+1];
    const prob_t *restrict trans_to_next = &transition[next_state];
    min_s = 0;
    min_p = llike_t[0] + trans_to_next[0 * n_states];
    for( s = 1; s < n_states; s++ ) {
      p = llike_t[s] + trans_to_next[s * n_states];
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

