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

  // Initialize with first observation and initial probabilities
  {
    const tok_t o0 = obs[0];
    const size_t emission_row_stride = (size_t)N_TOKENS;
    for( s = 0; s < (state_t)N_STATES; s++ ) {
      const size_t base = (size_t)s * emission_row_stride + (size_t)o0;
      llike[0][s] = init[s] + emission[base];
    }
  }

  // Iteratively compute the probabilities over time
  {
    const size_t trans_row_stride = (size_t)N_STATES;
    const size_t emission_row_stride = (size_t)N_TOKENS;

    for( t = 1; t < (step_t)N_OBS; t++ ) {
      const tok_t ot = obs[t];

      for( curr = 0; curr < (state_t)N_STATES; curr++ ) {
        const size_t emission_base = (size_t)curr * emission_row_stride + (size_t)ot;
        const prob_t emit_val = emission[emission_base];

        // prev = 0 case
        {
          const size_t trans_index0 = (size_t)0 * trans_row_stride + (size_t)curr;
          min_p = llike[t-1][0] + transition[trans_index0] + emit_val;
        }

        for( prev = 1; prev < (state_t)N_STATES; prev++ ) {
          const size_t trans_index = (size_t)prev * trans_row_stride + (size_t)curr;
          p = llike[t-1][prev] + transition[trans_index] + emit_val;
          if( p < min_p ) {
            min_p = p;
          }
        }

        llike[t][curr] = min_p;
      }
    }
  }

  // Identify end state
  {
    const step_t last_t = (step_t)(N_OBS - 1);
    min_s = 0;
    min_p = llike[last_t][0];
    for( s = 1; s < (state_t)N_STATES; s++ ) {
      p = llike[last_t][s];
      if( p < min_p ) {
        min_p = p;
        min_s = s;
      }
    }
    path[last_t] = min_s;
  }

  // Backtrack to recover full path
  {
    const size_t trans_row_stride = (size_t)N_STATES;
    for( t = (step_t)N_OBS - 2; t >= 0; t-- ) {
      const state_t next_state = path[t+1];
      const size_t trans_col = (size_t)next_state;

      min_s = 0;
      {
        const size_t trans_index0 = (size_t)0 * trans_row_stride + trans_col;
        min_p = llike[t][0] + transition[trans_index0];
      }

      for( s = 1; s < (state_t)N_STATES; s++ ) {
        const size_t trans_index = (size_t)s * trans_row_stride + trans_col;
        p = llike[t][s] + transition[trans_index];
        if( p < min_p ) {
          min_p = p;
          min_s = s;
        }
      }
      path[t] = min_s;
      if( t == 0 ) {
        break;
      }
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  viterbi_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                             (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
  return 0;
}

