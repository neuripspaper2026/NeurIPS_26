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

  const int n_states = N_STATES;
  const int n_obs = N_OBS;
  const int n_tokens = N_TOKENS;

  // Initialize with first observation and initial probabilities
  {
    const tok_t obs0 = obs[0];
    const prob_t *restrict init_ptr = init;
    const prob_t *restrict emission_ptr = emission;
    prob_t *restrict llike_row0 = llike[0];

    #pragma omp parallel for if(n_states > 8) default(none) shared(llike_row0, init_ptr, emission_ptr, obs0, n_states, n_tokens)
    for( state_t s_local = 0; s_local < n_states; s_local++ ) {
      llike_row0[s_local] = init_ptr[s_local] + emission_ptr[(size_t)s_local * n_tokens + obs0];
    }
  }

  // Iteratively compute the probabilities over time
  for( t = 1; t < n_obs; t++ ) {
    const tok_t obs_t = obs[t];
    const prob_t *restrict transition_ptr = transition;
    const prob_t *restrict emission_ptr = emission;
    prob_t *restrict llike_prev = llike[t-1];
    prob_t *restrict llike_curr = llike[t];

    #pragma omp parallel for if(n_states > 1) default(none) shared(llike_prev, llike_curr, transition_ptr, emission_ptr, obs_t, n_states, n_tokens)
    for( state_t curr_local = 0; curr_local < n_states; curr_local++ ) {
      prob_t min_p_local;
      prob_t p_local;

      // prev = 0 case
      {
        const state_t prev_local = 0;
        const prob_t lprev = llike_prev[prev_local];
        const prob_t trans = transition_ptr[(size_t)prev_local * n_states + curr_local];
        const prob_t emis = emission_ptr[(size_t)curr_local * n_tokens + obs_t];
        min_p_local = lprev + trans + emis;
      }

      for( state_t prev_local = 1; prev_local < n_states; prev_local++ ) {
        const prob_t lprev = llike_prev[prev_local];
        const prob_t trans = transition_ptr[(size_t)prev_local * n_states + curr_local];
        const prob_t emis = emission_ptr[(size_t)curr_local * n_tokens + obs_t];
        p_local = lprev + trans + emis;
        if( p_local < min_p_local ) {
          min_p_local = p_local;
        }
      }
      llike_curr[curr_local] = min_p_local;
    }
  }

  // Identify end state
  {
    prob_t *restrict llike_last = llike[n_obs-1];
    min_s = 0;
    min_p = llike_last[min_s];

    #pragma omp parallel for if(n_states > 1) default(none) shared(llike_last, n_states) reduction(min:min_p)
    for( state_t s_local = 0; s_local < n_states; s_local++ ) {
      prob_t p_local = llike_last[s_local];
      if( p_local < min_p ) {
        min_p = p_local;
      }
    }

    for( s = 0; s < n_states; s++ ) {
      if( llike_last[s] == min_p ) {
        min_s = s;
        break;
      }
    }
  }
  path[n_obs-1] = min_s;

  // Backtrack to recover full path (serial; depends on previous step)
  for( t = n_obs-2; t >= 0; t-- ) {
    prob_t *restrict llike_row = llike[t];
    const state_t next_state = path[t+1];
    const prob_t *restrict transition_ptr = transition;

    min_s = 0;
    min_p = llike_row[min_s] + transition_ptr[(size_t)min_s * n_states + next_state];

    for( s = 1; s < n_states; s++ ) {
      p = llike_row[s] + transition_ptr[(size_t)s * n_states + next_state];
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

