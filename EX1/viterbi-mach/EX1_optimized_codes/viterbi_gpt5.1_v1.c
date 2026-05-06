#include <time.h>
#include "../viterbi.h"

static double viterbi_kernel_time_acc = 0.0;

void reset_viterbi_kernel_time(void) { viterbi_kernel_time_acc = 0.0; }
double get_viterbi_kernel_time(void) { return viterbi_kernel_time_acc; }

int viterbi( tok_t obs[N_OBS], prob_t init[N_STATES], prob_t transition[N_STATES*N_STATES], prob_t emission[N_STATES*N_TOKENS], state_t path[N_OBS] )
{
  /* Move large array to static storage to avoid stack pressure */
  static prob_t llike[N_OBS][N_STATES];
  step_t t;
  state_t prev, curr;
  prob_t min_p, p;
  state_t min_s, s;

  /* Local pointers to frequently used arrays to help the compiler optimize */
  prob_t *restrict const transition_ptr = transition;
  prob_t *restrict const emission_ptr   = emission;
  tok_t  *restrict const obs_ptr        = obs;
  prob_t *restrict const init_ptr       = init;

  struct timespec kernel_start, kernel_end;

  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  {
    const tok_t o0 = obs_ptr[0];
    prob_t *restrict const llike_row0 = llike[0];

    for( s = 0; s < N_STATES; ++s ) {
      const prob_t init_v = init_ptr[s];
      const prob_t emit_v = emission_ptr[(size_t)s * N_TOKENS + o0];
      llike_row0[s] = init_v + emit_v;
    }
  }

  for( t = 1; t < N_OBS; ++t ) {
    const tok_t ot = obs_ptr[t];
    prob_t *restrict const llike_prev = llike[t-1];
    prob_t *restrict const llike_curr = llike[t];

    for( curr = 0; curr < N_STATES; ++curr ) {
      const size_t emit_idx  = (size_t)curr * N_TOKENS + ot;
      const prob_t emit_term = emission_ptr[emit_idx];

      prev = 0;
      {
        const size_t trans_idx0 = (size_t)0 * N_STATES + curr;
        min_p = llike_prev[0] + transition_ptr[trans_idx0] + emit_term;
      }

      for( prev = 1; prev < N_STATES; ++prev ) {
        const size_t trans_idx = (size_t)prev * N_STATES + curr;
        p = llike_prev[prev] + transition_ptr[trans_idx] + emit_term;
        if( p < min_p ) {
          min_p = p;
        }
      }
      llike_curr[curr] = min_p;
    }
  }

  {
    prob_t *restrict const llike_last = llike[N_OBS-1];
    min_s = 0;
    min_p = llike_last[min_s];

    for( s = 1; s < N_STATES; ++s ) {
      p = llike_last[s];
      if( p < min_p ) {
        min_p = p;
        min_s = s;
      }
    }
    path[N_OBS-1] = min_s;
  }

  for( t = N_OBS-2; t >= 0; --t ) {
    const state_t next_state = path[t+1];
    const size_t trans_col_offset = (size_t)next_state; /* used in indexing */
    prob_t *restrict const llike_row = llike[t];

    min_s = 0;
    {
      const size_t trans_idx0 = (size_t)0 * N_STATES + trans_col_offset;
      min_p = llike_row[0] + transition_ptr[trans_idx0];
    }

    for( s = 1; s < N_STATES; ++s ) {
      const size_t trans_idx = (size_t)s * N_STATES + trans_col_offset;
      p = llike_row[s] + transition_ptr[trans_idx];
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

