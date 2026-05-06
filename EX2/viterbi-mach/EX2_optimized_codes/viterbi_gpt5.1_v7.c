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
  prob_t llike[N_OBS][N_STATES];
  step_t t;
  state_t prev, curr;
  prob_t min_p, p;
  state_t min_s, s;
  struct timespec kernel_start, kernel_end;

  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  /* Precompute frequently used products for better cache locality */
  prob_t *trans = transition;
  prob_t *emis  = emission;

  /* Initialize with first observation and initial probabilities */
  {
    const tok_t o0 = obs[0];
    const int o0_offset = (int)o0;
    for( s = 0; s < N_STATES; s++ ) {
      llike[0][s] = init[s] + emis[s * N_TOKENS + o0_offset];
    }
  }

  /* Iteratively compute the probabilities over time */
  for( t = 1; t < N_OBS; t++ ) {
    const tok_t ot = obs[t];
    const int ot_offset = (int)ot;

    /* Parallelize over current state dimension; each iteration is independent */
    #ifdef _OPENMP
    #pragma omp parallel for private(prev, min_p, p) schedule(static)
    #endif
    for( curr = 0; curr < N_STATES; curr++ ) {
      const prob_t *restrict llike_prev = &llike[t-1][0];
      const prob_t *restrict trans_col  = &trans[curr]; /* accessed as trans[prev*N_STATES + curr] */
      const prob_t emit_val = emis[curr * N_TOKENS + ot_offset];

      prob_t local_min_p = llike_prev[0] + trans_col[0 * N_STATES] + emit_val;

      for( prev = 1; prev < N_STATES; prev++ ) {
        p = llike_prev[prev] + trans_col[prev * N_STATES] + emit_val;
        if( p < local_min_p ) {
          local_min_p = p;
        }
      }
      llike[t][curr] = local_min_p;
    }
  }

  /* Identify end state */
  min_s = 0;
  min_p = llike[N_OBS-1][min_s];
  for( s = 1; s < N_STATES; s++ ) {
    p = llike[N_OBS-1][s];
    if( p < min_p ) {
      min_p = p;
      min_s = s;
    }
  }
  path[N_OBS-1] = min_s;

  /* Backtrack to recover full path */
  for( t = N_OBS-2; t >= 0; t-- ) {
    const state_t next_state = path[t+1];
    const prob_t *restrict llike_t = &llike[t][0];
    const prob_t *restrict trans_row0 = &trans[0 * N_STATES + next_state];

    min_s = 0;
    min_p = llike_t[0] + trans_row0[0];

    for( s = 1; s < N_STATES; s++ ) {
      const prob_t *restrict trans_row = &trans[s * N_STATES + next_state];
      p = llike_t[s] + *trans_row;
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

