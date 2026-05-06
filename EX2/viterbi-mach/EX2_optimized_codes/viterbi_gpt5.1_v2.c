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
  /* Use static to avoid large stack frame and enable better locality */
  static prob_t llike[N_OBS][N_STATES];
  step_t t;
  state_t prev, curr;
  prob_t min_p, p;
  state_t min_s, s;
  struct timespec kernel_start, kernel_end;

  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  /* Precompute token index offsets for emissions to reduce address arithmetic */
  uint32_t emis_offset[N_OBS];
  for (t = 0; t < N_OBS; ++t) {
    emis_offset[t] = (uint32_t)obs[t] * (uint32_t)N_STATES;
  }

  /* Initialize with first observation and initial probabilities */
  {
    const uint32_t eoff0 = emis_offset[0];
    /* Parallelize over states if OpenMP available, encourage vectorization */
    #pragma omp parallel for if(N_STATES > 1) private(s) schedule(static)
    for (s = 0; s < N_STATES; s++) {
      llike[0][s] = init[s] + emission[eoff0 + s];
    }
  }

  /* Iteratively compute the probabilities over time */
  for (t = 1; t < N_OBS; t++) {
    const uint32_t eoff = emis_offset[t];

    /* Parallel over current state; each iteration is independent */
    #pragma omp parallel for if(N_STATES > 1) private(curr, prev, min_p, p) schedule(static)
    for (curr = 0; curr < N_STATES; curr++) {
      /* Unroll first iteration of prev loop */
      prob_t best = llike[t-1][0] +
                    transition[(state_t)0 * N_STATES + curr] +
                    emission[eoff + curr];

      /* Pointer to previous-time likelihood row for better locality */
      prob_t const *restrict llike_prev = llike[t-1];
      prob_t const *restrict trans_col  = &transition[curr]; /* we index prev*N_STATES+curr */

      /* Scan over all previous states to find minimum */
      for (prev = 1; prev < N_STATES; prev++) {
        p = llike_prev[prev] +
            trans_col[prev * (step_t)N_STATES] +
            emission[eoff + curr];
        if (p < best) {
          best = p;
        }
      }
      llike[t][curr] = best;
    }
  }

  /* Identify end state */
  min_s = 0;
  min_p = llike[N_OBS-1][min_s];
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
    min_s = 0;
    min_p = llike[t][min_s] + transition[min_s * N_STATES + path[t+1]];
    for (s = 1; s < N_STATES; s++) {
      p = llike[t][s] + transition[s * N_STATES + path[t+1]];
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

