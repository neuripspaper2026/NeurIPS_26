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

  // Precompute and store emission indices for each observation
  int emm_idx[N_OBS];
  for (t = 0; t < n_obs; ++t) {
    emm_idx[t] = (int)obs[t];
  }

  // Initialize with first observation and initial probabilities
  {
    const int tok0 = emm_idx[0];
    const int tok0_offset = tok0; // emission[s * n_tokens + tok0]
    for (s = 0; s < n_states; ++s) {
      llike[0][s] = init[s] + emission[(int)s * n_tokens + tok0_offset];
    }
  }

  // Iteratively compute the probabilities over time
  for (t = 1; t < n_obs; ++t) {
    const int tok  = emm_idx[t];
    const int tok_offset = tok; // emission[curr * n_tokens + tok]
    const prob_t *restrict llike_prev = llike[t-1];
    prob_t *restrict llike_curr = llike[t];
    for (curr = 0; curr < n_states; ++curr) {
      const int trans_col_offset = (int)curr; // transition[prev * n_states + curr]
      const prob_t emit_val = emission[(int)curr * n_tokens + tok_offset];

      prev = 0;
      prob_t min_val = llike_prev[0] +
                       transition[(int)0 * n_states + trans_col_offset] +
                       emit_val;

      for (prev = 1; prev < n_states; ++prev) {
        p = llike_prev[prev] +
            transition[(int)prev * n_states + trans_col_offset] +
            emit_val;
        if (p < min_val) {
          min_val = p;
        }
      }
      llike_curr[curr] = min_val;
    }
  }

  // Identify end state
  {
    const prob_t *restrict llike_last = llike[n_obs-1];
    min_s = 0;
    min_p = llike_last[min_s];
    for (s = 1; s < n_states; ++s) {
      p = llike_last[s];
      if (p < min_p) {
        min_p = p;
        min_s = s;
      }
    }
    path[n_obs-1] = min_s;
  }

  // Backtrack to recover full path
  for (t = n_obs-2; t >= 0; --t) {
    prob_t *restrict llike_t = llike[t];
    const state_t next_state = path[t+1];
    const int trans_row_offset_next = (int)next_state;
    min_s = 0;
    min_p = llike_t[0] + transition[(int)0 * n_states + trans_row_offset_next];
    for (s = 1; s < n_states; ++s) {
      p = llike_t[s] + transition[(int)s * n_states + trans_row_offset_next];
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

