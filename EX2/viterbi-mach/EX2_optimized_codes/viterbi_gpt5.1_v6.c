#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../viterbi.h"

static double viterbi_kernel_time_acc = 0.0;

void reset_viterbi_kernel_time(void) { viterbi_kernel_time_acc = 0.0; }
double get_viterbi_kernel_time(void) { return viterbi_kernel_time_acc; }

int viterbi( tok_t obs[N_OBS], prob_t init[N_STATES], prob_t transition[N_STATES*N_STATES], prob_t emission[N_STATES*N_TOKENS], state_t path[N_OBS] )
{
  /* Use static extents so compiler can better optimize (vectorize/unroll) */
  prob_t llike[N_OBS][N_STATES];

  step_t t;
  state_t prev, curr;
  prob_t min_p, p;
  state_t min_s, s;

  struct timespec kernel_start, kernel_end;

  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  /* Precompute strides (avoid repeated multiplies in inner loops) */
  const int n_states  = N_STATES;
  const int n_obs     = N_OBS;
  const int n_tokens  = N_TOKENS;
  const int stride_em = n_tokens;
  const int stride_tr = n_states;

  /* Initialize with first observation and initial probabilities */
  {
    const tok_t o0 = obs[0];
    const int em_base = o0;
    /* Independent across states: parallelizable */
    #pragma omp parallel for private(s) if(n_states > 1)
    for (s = 0; s < n_states; s++) {
      llike[0][s] = init[s] + emission[s*stride_em + em_base];
    }
  }

  /* Iteratively compute the probabilities over time */
  for (t = 1; t < n_obs; t++) {
    const tok_t ot = obs[t];
    const int em_off = ot;
    const int t_prev = t - 1;

    /* Each current state is independent in this timestep */
    #pragma omp parallel for private(curr, prev, min_p, p) if(n_states > 1)
    for (curr = 0; curr < n_states; curr++) {
      const prob_t *restrict llike_prev = &llike[t_prev][0];
      const prob_t *restrict trans_col  = &transition[curr]; /* accessed as trans_col[prev*stride_tr] */
      const prob_t emit_val = emission[curr*stride_em + em_off];

      /* Initialize with prev = 0 */
      prev  = 0;
      min_p = llike_prev[0] + trans_col[0*stride_tr] + emit_val;

      /* Scan all previous states, find minimum */
      for (prev = 1; prev < n_states; prev++) {
        p = llike_prev[prev] + trans_col[prev*stride_tr] + emit_val;
        if (p < min_p) {
          min_p = p;
        }
      }

      llike[t][curr] = min_p;
    }
  }

  /* Identify end state */
  min_s = 0;
  min_p = llike[n_obs-1][min_s];

  #pragma omp parallel for private(s,p) reduction(min:min_p) if(n_states > 1)
  for (s = 1; s < n_states; s++) {
    p = llike[n_obs-1][s];
    if (p < min_p) {
      min_p = p;
    }
  }

  /* After reduction we need the index of the min; recompute once serially */
  for (s = 0; s < n_states; s++) {
    if (llike[n_obs-1][s] == min_p) {
      min_s = s;
      break;
    }
  }

  path[n_obs-1] = min_s;

  /* Backtrack to recover full path (sequential dependence in time) */
  for (t = n_obs-2; t >= 0; t--) {
    const int t_next = t + 1;
    const state_t next_state = path[t_next];

    min_s = 0;
    min_p = llike[t][0] + transition[0*stride_tr + next_state];

    for (s = 1; s < n_states; s++) {
      p = llike[t][s] + transition[s*stride_tr + next_state];
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

