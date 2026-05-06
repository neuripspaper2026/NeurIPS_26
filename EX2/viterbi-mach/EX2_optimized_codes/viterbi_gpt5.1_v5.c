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
  /* Keep llike aligned and contiguous for better cache and vectorization */
  prob_t llike[N_OBS][N_STATES];
  step_t t;
  state_t prev, curr;
  prob_t min_p, p;
  state_t min_s, s;
  struct timespec kernel_start, kernel_end;

  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  /* Precompute emission index stride for cache friendliness */
  const int states_stride = N_STATES;
  const int tokens_stride = N_TOKENS;

  /* Initialize with first observation and initial probabilities */
  {
    const tok_t o0 = obs[0];
    const int em_base = o0; /* emission[s * N_TOKENS + o0] */
    #pragma omp simd
    L_init: for( s = 0; s < N_STATES; s++ ) {
      llike[0][s] = init[s] + emission[s * tokens_stride + em_base];
    }
  }

  /* Iteratively compute the probabilities over time */
  L_timestep: for( t = 1; t < N_OBS; t++ ) {
    const tok_t ot = obs[t];
    const int em_col = ot; /* column index in emission matrix */
    prob_t *restrict llike_prev = llike[t-1];
    prob_t *restrict llike_cur  = llike[t];

    /* Parallelize across current states; each iteration is independent */
    #ifdef _OPENMP
    #pragma omp parallel for private(prev, min_p, p) schedule(static)
    #endif
    L_curr_state: for( curr = 0; curr < N_STATES; curr++ ) {
      const int trans_col_offset = curr; /* transition[prev * N_STATES + curr] */
      const int em_idx = curr * tokens_stride + em_col;
      const prob_t em_val = emission[em_idx];

      /* Initialize with prev = 0 */
      min_p = llike_prev[0] +
              transition[0 * states_stride + trans_col_offset] +
              em_val;

      /* Compute minimal path probability over all previous states */
      #pragma omp simd reduction(min:min_p)
      L_prev_state: for( prev = 1; prev < N_STATES; prev++ ) {
        p = llike_prev[prev] +
            transition[prev * states_stride + trans_col_offset] +
            em_val;
        if( p < min_p ) {
          min_p = p;
        }
      }
      llike_cur[curr] = min_p;
    }
  }

  /* Identify end state (serial reduction over last timestep) */
  {
    prob_t *restrict llike_last = llike[N_OBS-1];
    min_s = 0;
    min_p = llike_last[min_s];
    L_end: for( s = 1; s < N_STATES; s++ ) {
      p = llike_last[s];
      if( p < min_p ) {
        min_p = p;
        min_s = s;
      }
    }
    path[N_OBS-1] = min_s;
  }

  /* Backtrack to recover full path (must be serial due to dependence) */
  L_backtrack: for( t = N_OBS - 2; t >= 0; t-- ) {
    const state_t next_state = path[t+1];
    const int trans_row_offset = next_state;
    min_s = 0;
    min_p = llike[t][min_s] + transition[min_s * states_stride + trans_row_offset];
    L_state: for( s = 1; s < N_STATES; s++ ) {
      p = llike[t][s] + transition[s * states_stride + trans_row_offset];
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

