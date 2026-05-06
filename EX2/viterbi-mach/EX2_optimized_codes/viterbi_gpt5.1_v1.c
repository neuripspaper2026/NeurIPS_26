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
  /* Use static allocation to avoid large stack frame and improve locality */
  static prob_t llike[N_OBS][N_STATES];
  step_t t;
  state_t prev, curr;
  prob_t min_p, p;
  state_t min_s, s;
  struct timespec kernel_start, kernel_end;

  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  /* Initialize with first observation and initial probabilities */
  {
    const tok_t obs0 = obs[0];
    const prob_t *restrict init_p      = init;
    const prob_t *restrict emission_p  = emission;
    const int stride_em = N_TOKENS;

    for( s = 0; s < N_STATES; s++ ) {
      llike[0][s] = init_p[s] + emission_p[(size_t)s*stride_em + obs0];
    }
  }

  /* Iteratively compute the probabilities over time */
  for( t = 1; t < N_OBS; t++ ) {
    const tok_t ob = obs[t];
    const prob_t *restrict trans_t    = transition;
    const prob_t *restrict emiss_t    = emission;
    const prob_t *restrict prev_llike = llike[t-1];
    prob_t *restrict curr_llike       = llike[t];

    /* Parallel over current states; inner reduction over previous states */
    #ifdef _OPENMP
    #pragma omp parallel for private(prev,min_p,p) schedule(static)
    #endif
    for( curr = 0; curr < N_STATES; curr++ ) {
      const prob_t emit_val = emiss_t[(size_t)curr * N_TOKENS + ob];

      /* Unroll first iteration to initialize min_p */
      prev  = 0;
      min_p = prev_llike[0] +
              trans_t[(size_t)0 * N_STATES + curr] +
              emit_val;

      for( prev = 1; prev < N_STATES; prev++ ) {
        p = prev_llike[prev] +
            trans_t[(size_t)prev * N_STATES + curr] +
            emit_val;
        if( p < min_p ) {
          min_p = p;
        }
      }
      curr_llike[curr] = min_p;
    }
  }

  /* Identify end state (serial; small cost) */
  min_s = 0;
  min_p = llike[N_OBS-1][0];
  {
    const prob_t *restrict last_llike = llike[N_OBS-1];
    for( s = 1; s < N_STATES; s++ ) {
      p = last_llike[s];
      if( p < min_p ) {
        min_p = p;
        min_s = s;
      }
    }
  }
  path[N_OBS-1] = min_s;

  /* Backtrack to recover full path (serial dependence on path) */
  for( t = N_OBS-2; t >= 0; t-- ) {
    const state_t next_state = path[t+1];
    const prob_t *restrict llike_t = llike[t];
    const prob_t *restrict trans_t = transition;

    min_s = 0;
    min_p = llike_t[0] + trans_t[(size_t)0 * N_STATES + next_state];

    for( s = 1; s < N_STATES; s++ ) {
      p = llike_t[s] + trans_t[(size_t)s * N_STATES + next_state];
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

