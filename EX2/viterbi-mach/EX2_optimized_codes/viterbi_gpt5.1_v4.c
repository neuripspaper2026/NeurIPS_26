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
  /* Keep llike as a fixed-size array to allow good cache layout and possible
     compiler vectorization; avoid dynamic allocation. */
  prob_t llike[N_OBS][N_STATES];

  step_t t;
  state_t prev, curr;
  prob_t min_p, p;
  state_t min_s, s;

  /* All probabilities are in -log space. (i.e.: P(x) => -log(P(x)) ) */
  struct timespec kernel_start, kernel_end;

  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  /* Initialize with first observation and initial probabilities.
     This loop is trivially parallelizable over states. */
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(llike, init, emission, obs) private(s)
#endif
  for (s = 0; s < (state_t)N_STATES; s++) {
    llike[0][s] = init[s] + emission[(size_t)s * (size_t)N_TOKENS + (size_t)obs[0]];
  }

  /* Iteratively compute the probabilities over time.
     Each time step depends only on the previous time step; we keep the
     outer time loop serial and parallelize the inner "curr" loop. */
  for (t = 1; t < (step_t)N_OBS; t++) {
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(llike, transition, emission, obs, t) private(curr, prev, min_p, p)
#endif
    for (curr = 0; curr < (state_t)N_STATES; curr++) {
      /* Compute likelihood HMM is in current state and where it came from. */
      prev  = 0;
      {
        const size_t base_prev0 = (size_t)prev * (size_t)N_STATES + (size_t)curr;
        const size_t base_emit  = (size_t)curr * (size_t)N_TOKENS + (size_t)obs[t];
        min_p = llike[t-1][prev] + transition[base_prev0] + emission[base_emit];
      }

      /* Manually unroll the inner loop a bit to improve ILP.
         Each thread works on its own (curr), so no data race. */
      for (prev = 1; prev + 3 < (state_t)N_STATES; prev += 4) {
        size_t base0 = (size_t)prev * (size_t)N_STATES + (size_t)curr;
        size_t base1 = base0 + (size_t)N_STATES;
        size_t base2 = base1 + (size_t)N_STATES;
        size_t base3 = base2 + (size_t)N_STATES;

        prob_t prev_ll0 = llike[t-1][prev];
        prob_t prev_ll1 = llike[t-1][(state_t)(prev + 1)];
        prob_t prev_ll2 = llike[t-1][(state_t)(prev + 2)];
        prob_t prev_ll3 = llike[t-1][(state_t)(prev + 3)];

        prob_t cand0 = prev_ll0 + transition[base0] + emission[(size_t)curr * (size_t)N_TOKENS + (size_t)obs[t]];
        prob_t cand1 = prev_ll1 + transition[base1] + emission[(size_t)curr * (size_t)N_TOKENS + (size_t)obs[t]];
        prob_t cand2 = prev_ll2 + transition[base2] + emission[(size_t)curr * (size_t)N_TOKENS + (size_t)obs[t]];
        prob_t cand3 = prev_ll3 + transition[base3] + emission[(size_t)curr * (size_t)N_TOKENS + (size_t)obs[t]];

        if (cand0 < min_p) min_p = cand0;
        if (cand1 < min_p) min_p = cand1;
        if (cand2 < min_p) min_p = cand2;
        if (cand3 < min_p) min_p = cand3;
      }

      for (; prev < (state_t)N_STATES; prev++) {
        size_t base_prev = (size_t)prev * (size_t)N_STATES + (size_t)curr;
        size_t base_emit = (size_t)curr * (size_t)N_TOKENS + (size_t)obs[t];
        p = llike[t-1][prev] + transition[base_prev] + emission[base_emit];
        if (p < min_p) {
          min_p = p;
        }
      }

      llike[t][curr] = min_p;
    }
  }

  /* Identify end state: parallel reduction on minimum over states. */
  min_s = 0;
  min_p = llike[N_OBS-1][min_s];

#ifdef _OPENMP
#pragma omp parallel for default(none) shared(llike) reduction(min:min_p) private(s) firstprivate(min_p)
#endif
  for (s = 1; s < (state_t)N_STATES; s++) {
    prob_t val = llike[N_OBS-1][s];
    if (val < min_p) {
      min_p = val;
    }
  }

  /* Recover the index of min_p; this loop is cheap and remains serial. */
  min_s = 0;
  min_p = llike[N_OBS-1][min_s];
  for (s = 1; s < (state_t)N_STATES; s++) {
    p = llike[N_OBS-1][s];
    if (p < min_p) {
      min_p = p;
      min_s = s;
    }
  }

  path[N_OBS-1] = min_s;

  /* Backtrack to recover full path.
     Each step depends on the next step's chosen state, so this loop
     is inherently serial. */
  for (t = (step_t)N_OBS - 2; t >= 0; t--) {
    min_s = 0;
    min_p = llike[t][min_s] + transition[(size_t)min_s * (size_t)N_STATES + (size_t)path[t+1]];

    for (s = 1; s < (state_t)N_STATES; s++) {
      size_t base_tr = (size_t)s * (size_t)N_STATES + (size_t)path[t+1];
      p = llike[t][s] + transition[base_tr];
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

