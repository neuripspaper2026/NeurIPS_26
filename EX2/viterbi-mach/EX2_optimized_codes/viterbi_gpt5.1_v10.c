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

  // Initialize with first observation and initial probabilities
  {
    const tok_t o0 = obs[0];
    const prob_t *restrict init_ptr = init;
    const prob_t *restrict emis_base = emission;
    prob_t *restrict l0 = llike[0];
#ifdef _OPENMP
#pragma omp parallel for if(N_STATES > 1) default(none) shared(init_ptr, emis_base, l0, o0) private(s)
#endif
    for( s = 0; s < N_STATES; s++ ) {
      l0[s] = init_ptr[s] + emis_base[(size_t)s * N_TOKENS + o0];
    }
  }

  // Iteratively compute the probabilities over time
  for( t = 1; t < N_OBS; t++ ) {
    const tok_t ot = obs[t];
    const prob_t *restrict emis_base = emission;
    prob_t *restrict l_prev = llike[t-1];
    prob_t *restrict l_curr = llike[t];
#ifdef _OPENMP
#pragma omp parallel for if(N_STATES > 1) default(none) shared(l_prev, l_curr, transition, emis_base, ot) private(curr, prev, min_p, p)
#endif
    for( curr = 0; curr < N_STATES; curr++ ) {
      const prob_t *restrict trans_col = &transition[(size_t)curr];
      const prob_t emis = emis_base[(size_t)curr * N_TOKENS + ot];

      prev = 0;
      min_p = l_prev[0] + trans_col[0 * (size_t)N_STATES] + emis;

      for( prev = 1; prev < N_STATES; prev++ ) {
        p = l_prev[prev] + trans_col[(size_t)prev * N_STATES] + emis;
        if( p < min_p ) {
          min_p = p;
        }
      }
      l_curr[curr] = min_p;
    }
  }

  // Identify end state
  {
    prob_t *restrict l_last = llike[N_OBS-1];
    min_s = 0;
    min_p = l_last[0];
    for( s = 1; s < N_STATES; s++ ) {
      p = l_last[s];
      if( p < min_p ) {
        min_p = p;
        min_s = s;
      }
    }
    path[N_OBS-1] = min_s;
  }

  // Backtrack to recover full path
  for( t = N_OBS - 2; t >= 0; t-- ) {
    prob_t *restrict l_t = llike[t];
    const state_t next_state = path[t+1];
    const prob_t *restrict trans_to_next = &transition[(size_t)next_state];
    min_s = 0;
    min_p = l_t[0] + trans_to_next[0 * (size_t)N_STATES];
    for( s = 1; s < N_STATES; s++ ) {
      p = l_t[s] + trans_to_next[(size_t)s * N_STATES];
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

