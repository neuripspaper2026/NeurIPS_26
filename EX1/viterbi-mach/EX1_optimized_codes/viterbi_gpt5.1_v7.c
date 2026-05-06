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

  /* Precompute observation emission index offsets to avoid repeated multiplies */
  int obs_idx[N_OBS];
  for (t = 0; t < N_OBS; t++) {
    obs_idx[t] = (int)obs[t] * N_STATES;
  }

  /* Precompute transition index offsets for each previous state */
  int trans_prev_off[N_STATES];
  for (prev = 0; prev < N_STATES; prev++) {
    trans_prev_off[prev] = (int)prev * N_STATES;
  }

  /* Initialize with first observation and initial probabilities */
  {
    int eo = obs_idx[0]; /* emission offset for first observation */
    for (s = 0; s < N_STATES; s++) {
      llike[0][s] = init[s] + emission[eo + s];
    }
  }

  /* Iteratively compute the probabilities over time */
  for (t = 1; t < N_OBS; t++) {
    int eo = obs_idx[t]; /* emission offset for current observation */
    for (curr = 0; curr < N_STATES; curr++) {
      const prob_t emit_val = emission[eo + curr];
      /* Handle prev = 0 separately to initialize min_p */
      prev = 0;
      min_p = llike[t-1][0] +
              transition[trans_prev_off[0] + curr] +
              emit_val;

      for (prev = 1; prev < N_STATES; prev++) {
        p = llike[t-1][prev] +
            transition[trans_prev_off[prev] + curr] +
            emit_val;
        if (p < min_p) {
          min_p = p;
        }
      }
      llike[t][curr] = min_p;
    }
  }

  /* Identify end state */
  min_s = 0;
  min_p = llike[N_OBS-1][0];
  for (s = 1; s < N_STATES; s++) {
    p = llike[N_OBS-1][s];
    if (p < min_p) {
      min_p = p;
      min_s = s;
    }
  }
  path[N_OBS-1] = min_s;

  /* Backtrack to recover full path */
  for (t = N_OBS-2; t >= 0; t--) {
    int path_next = path[t+1];
    int trans_next_off = path_next; /* used as column index */

    min_s = 0;
    min_p = llike[t][0] + transition[trans_next_off];
    for (s = 1; s < N_STATES; s++) {
      p = llike[t][s] + transition[s*N_STATES + trans_next_off];
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

