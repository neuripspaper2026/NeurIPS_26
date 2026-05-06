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

  // pointers and constants for faster index calculations
  const int n_states = N_STATES;
  const int n_tokens = N_TOKENS;
  const int n_obs    = N_OBS;

  // Initialize with first observation and initial probabilities
  {
    const tok_t o0 = obs[0];
    const int o0_offset = (int)o0;
    prob_t *restrict llike_row0 = llike[0];
    const prob_t *restrict init_ptr = init;
    const prob_t *restrict emission_ptr = emission;

    for (s = 0; s < n_states; s++) {
      llike_row0[s] = init_ptr[s] + emission_ptr[s * n_tokens + o0_offset];
    }
  }

  // Iteratively compute the probabilities over time
  for (t = 1; t < n_obs; t++) {
    const tok_t ot = obs[t];
    const int ot_offset = (int)ot;
    prob_t *restrict llike_curr = llike[t];
    prob_t *restrict llike_prev = llike[t - 1];
    const prob_t *restrict emission_ptr = emission;
    const prob_t *restrict transition_ptr = transition;

    for (curr = 0; curr < n_states; curr++) {
      const int curr_offset_emission = curr * n_tokens + ot_offset;
      const prob_t emit_curr = emission_ptr[curr_offset_emission];

      // prev = 0 case initialized separately
      prev = 0;
      prob_t min_local = llike_prev[0] +
                         transition_ptr[(size_t)0 * n_states + curr] +
                         emit_curr;

      for (prev = 1; prev < n_states; prev++) {
        p = llike_prev[prev] +
            transition_ptr[(size_t)prev * n_states + curr] +
            emit_curr;
        if (p < min_local) {
          min_local = p;
        }
      }
      llike_curr[curr] = min_local;
    }
  }

  // Identify end state
  {
    prob_t *restrict llike_last = llike[n_obs - 1];
    min_s = 0;
    min_p = llike_last[min_s];
    for (s = 1; s < n_states; s++) {
      p = llike_last[s];
      if (p < min_p) {
        min_p = p;
        min_s = s;
      }
    }
    path[n_obs - 1] = min_s;
  }

  // Backtrack to recover full path
  for (t = n_obs - 2; t >= 0; t--) {
    prob_t *restrict llike_row = llike[t];
    const state_t next_state = path[t + 1];
    const int next_state_offset = (int)next_state;
    const prob_t *restrict transition_ptr = transition;

    min_s = 0;
    min_p = llike_row[0] + transition_ptr[(size_t)0 * n_states + next_state_offset];

    for (s = 1; s < n_states; s++) {
      p = llike_row[s] + transition_ptr[(size_t)s * n_states + next_state_offset];
      if (p < min_p) {
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

