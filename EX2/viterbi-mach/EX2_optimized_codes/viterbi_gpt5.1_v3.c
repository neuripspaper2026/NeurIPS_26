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

  /* Precompute base pointer for transition (row-major: prev * N_STATES + curr) */
  prob_t *restrict transition_base = transition;
  prob_t *restrict emission_base   = emission;

  /* Initialize with first observation and initial probabilities */
  {
    const tok_t o0 = obs[0];
    const int o0_offset = (int)o0; /* emission index offset */
    /* Parallelize state initialization; each iteration is independent */
    #pragma omp parallel for if(N_STATES > 8) default(none) shared(llike, init, emission_base, o0_offset)
    for (state_t s_local = 0; s_local < N_STATES; s_local++) {
      llike[0][s_local] = init[s_local] + emission_base[(size_t)s_local * N_TOKENS + o0_offset];
    }
  }

  /* Iteratively compute the probabilities over time */
  for (t = 1; t < N_OBS; t++) {
    const tok_t ot = obs[t];
    const int ot_offset = (int)ot;

    /* Parallelize over current state; each curr uses disjoint llike[t][curr] */
    #pragma omp parallel for if(N_STATES > 1) default(none) private(prev, p, min_p) shared(t, llike, transition_base, emission_base, ot_offset)
    for (state_t curr_local = 0; curr_local < N_STATES; curr_local++) {
      /* Load emission once per (t,curr) */
      const prob_t emit_val = emission_base[(size_t)curr_local * N_TOKENS + ot_offset];

      /* Compute likelihood HMM is in current state and where it came from. */
      prev = 0;
      prob_t best = llike[t-1][0] +
                    transition_base[(size_t)0 * N_STATES + curr_local] +
                    emit_val;

      for (prev = 1; prev < N_STATES; prev++) {
        p = llike[t-1][prev] +
            transition_base[(size_t)prev * N_STATES + curr_local] +
            emit_val;
        if (p < best) {
          best = p;
        }
      }
      llike[t][curr_local] = best;
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

  /* Backtrack to recover full path.
     This loop is inherently sequential; hoist invariant path index term. */
  for (t = N_OBS-2; t >= 0; t--) {
    const state_t next_state = path[t+1];
    const size_t next_col_offset = (size_t)next_state; /* column in transition */
    min_s = 0;
    min_p = llike[t][0] + transition_base[(size_t)0 * N_STATES + next_col_offset];
    for (s = 1; s < N_STATES; s++) {
      p = llike[t][s] + transition_base[(size_t)s * N_STATES + next_col_offset];
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

