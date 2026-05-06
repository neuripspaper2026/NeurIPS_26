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
  /* Use static extent for better locality; allocate as single block */
  prob_t llike[N_OBS][N_STATES];
  step_t t;
  state_t prev, curr;
  prob_t min_p, p;
  state_t min_s, s;
  struct timespec kernel_start, kernel_end;

  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  /* Hoist frequently used strides into locals */
  const step_t n_states  = N_STATES;
  const step_t n_tokens  = N_TOKENS;
  const step_t n_obs     = N_OBS;
  const step_t stride_tr = N_STATES;
  const step_t stride_em = N_TOKENS;

  /* Initialize with first observation and initial probabilities */
  {
    const tok_t o0 = obs[0];
    const step_t base_em = (step_t)o0; /* column index for emission */
    #pragma omp simd
    L_init: for( s = 0; s < n_states; s++ ) {
      llike[0][s] = init[s] + emission[(step_t)s * stride_em + base_em];
    }
  }

  /* Iteratively compute the probabilities over time */
  L_timestep: for( t = 1; t < n_obs; t++ ) {
    const tok_t ot      = obs[t];
    const step_t base_em_col = (step_t)ot;
    const prob_t *restrict llike_prev = llike[t-1];
    prob_t *restrict llike_curr       = llike[t];

    /* Parallelize across current states; each iteration is independent */
    #ifdef _OPENMP
    #pragma omp parallel for private(prev, min_p, p) schedule(static)
    #endif
    L_curr_state: for( curr = 0; curr < n_states; curr++ ) {
      const step_t curr_idx_em = (step_t)curr * stride_em + base_em_col;
      const prob_t emit_val    = emission[curr_idx_em];

      /* Unroll first iteration of prev loop */
      prev  = 0;
      min_p = llike_prev[0] +
              transition[(step_t)0 * stride_tr + curr] +
              emit_val;

      /* Remaining prev states */
      L_prev_state: for( prev = 1; prev < n_states; prev++ ) {
        p = llike_prev[prev] +
            transition[(step_t)prev * stride_tr + curr] +
            emit_val;
        if( p < min_p ) {
          min_p = p;
        }
      }
      llike_curr[curr] = min_p;
    }
  }

  /* Identify end state */
  {
    prob_t *restrict llike_last = llike[n_obs-1];
    min_s = 0;
    min_p = llike_last[0];

    #pragma omp simd reduction(min:min_p)
    L_end: for( s = 1; s < n_states; s++ ) {
      p = llike_last[s];
      if( p < min_p ) {
        min_p = p;
        min_s = s;
      }
    }
    /* second pass to recover corresponding state to min_p */
    for( s = 0; s < n_states; s++ ) {
      if( llike_last[s] == min_p ) {
        min_s = s;
        break;
      }
    }
    path[n_obs-1] = min_s;
  }

  /* Backtrack to recover full path
     (sequential due to data dependence between timesteps) */
  L_backtrack: for( t = n_obs-2; t >= 0; t-- ) {
    const state_t next_state = path[t+1];
    const step_t base_tr_col = (step_t)next_state;
    prob_t *restrict llike_t = llike[t];

    min_s = 0;
    min_p = llike_t[0] + transition[(step_t)0 * stride_tr + base_tr_col];

    L_state: for( s = 1; s < n_states; s++ ) {
      p = llike_t[s] + transition[(step_t)s * stride_tr + base_tr_col];
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

