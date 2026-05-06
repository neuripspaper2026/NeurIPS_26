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

  /* Precompute row pointers to improve cache locality and reduce indexing cost */
  prob_t *trans_rows[N_STATES];
  prob_t *emit_rows[N_STATES];
  for (s = 0; s < N_STATES; ++s) {
    trans_rows[s] = &transition[(size_t)s * (size_t)N_STATES];
    emit_rows[s]  = &emission[(size_t)s * (size_t)N_TOKENS];
  }

  /* Initialize with first observation and initial probabilities */
  {
    const tok_t o0 = obs[0];
    for (s = 0; s < N_STATES; ++s) {
      llike[0][s] = init[s] + emit_rows[s][o0];
    }
  }

  /* Iteratively compute the probabilities over time */
  for (t = 1; t < N_OBS; ++t) {
    const tok_t ot = obs[t];
    prob_t *llike_prev = llike[t - 1];
    prob_t *llike_curr = llike[t];

    for (curr = 0; curr < N_STATES; ++curr) {
      prob_t *trans_col_base = &trans_rows[0][curr];
      prob_t emit_val = emit_rows[curr][ot];

      prev = 0;
      prob_t best = llike_prev[0] + trans_col_base[0] + emit_val;

      for (prev = 1; prev < N_STATES; ++prev) {
        prob_t cand = llike_prev[prev] +
                      trans_rows[prev][curr] +
                      emit_val;
        if (cand < best) {
          best = cand;
        }
      }
      llike_curr[curr] = best;
    }
  }

  /* Identify end state */
  {
    prob_t *llike_last = llike[N_OBS - 1];
    min_s = 0;
    min_p = llike_last[0];
    for (s = 1; s < N_STATES; ++s) {
      p = llike_last[s];
      if (p < min_p) {
        min_p = p;
        min_s = s;
      }
    }
    path[N_OBS - 1] = min_s;
  }

  /* Backtrack to recover full path */
  for (t = N_OBS - 2; t >= 0; --t) {
    prob_t *llike_t = llike[t];
    const state_t next_state = path[t + 1];

    min_s = 0;
    min_p = llike_t[0] + trans_rows[0][next_state];

    for (s = 1; s < N_STATES; ++s) {
      p = llike_t[s] + trans_rows[s][next_state];
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

