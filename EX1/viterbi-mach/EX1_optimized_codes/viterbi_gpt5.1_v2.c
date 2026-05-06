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
  const step_t n_obs = N_OBS;
  const state_t n_states = N_STATES;
  const tok_t n_tokens = N_TOKENS;
  const tok_t * const __restrict obs_local = obs;
  prob_t * const __restrict init_local = init;
  prob_t * const __restrict transition_local = transition;
  prob_t * const __restrict emission_local = emission;
  state_t * const __restrict path_local = path;
  // All probabilities are in -log space. (i.e.: P(x) => -log(P(x)) )
  struct timespec kernel_start, kernel_end;

  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  // Initialize with first observation and initial probabilities
  {
    const tok_t o0 = obs_local[0];
    const prob_t * const __restrict emission_o0 = &emission_local[o0];
    for( s = 0; s < n_states; s++ ) {
      llike[0][s] = init_local[s] + emission_o0[s * n_tokens];
    }
  }

  // Iteratively compute the probabilities over time
  for( t = 1; t < n_obs; t++ ) {
    const tok_t ot = obs_local[t];
    const prob_t * const __restrict emission_ot = &emission_local[ot];
    const step_t t_prev = t - 1;
    for( curr = 0; curr < n_states; curr++ ) {
      const prob_t * const __restrict trans_col = &transition_local[curr];
      const prob_t emit_val = emission_ot[curr * n_tokens];
      prev = 0;
      prob_t best = llike[t_prev][0] + trans_col[0 * n_states] + emit_val;
      for( prev = 1; prev < n_states; prev++ ) {
        p = llike[t_prev][prev] + trans_col[prev * n_states] + emit_val;
        if( p < best ) {
          best = p;
        }
      }
      llike[t][curr] = best;
    }
  }

  // Identify end state
  {
    const step_t t_last = n_obs - 1;
    min_s = 0;
    min_p = llike[t_last][0];
    for( s = 1; s < n_states; s++ ) {
      p = llike[t_last][s];
      if( p < min_p ) {
        min_p = p;
        min_s = s;
      }
    }
    path_local[t_last] = min_s;
  }

  // Backtrack to recover full path
  {
    step_t tt = n_obs - 2;
    for( ; tt >= 0; tt-- ) {
      const state_t next_state = path_local[tt + 1];
      const prob_t * const __restrict trans_to_next = &transition_local[next_state];
      min_s = 0;
      min_p = llike[tt][0] + trans_to_next[0 * n_states];
      for( s = 1; s < n_states; s++ ) {
        p = llike[tt][s] + trans_to_next[s * n_states];
        if( p < min_p ) {
          min_p = p;
          min_s = s;
        }
      }
      path_local[tt] = min_s;
      if (tt == 0) {
        break;
      }
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  viterbi_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                             (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
  return 0;
}

