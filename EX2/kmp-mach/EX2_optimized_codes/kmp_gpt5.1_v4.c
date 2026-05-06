#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../kmp.h"

static double kmp_kernel_time_acc = 0.0;

void reset_kmp_kernel_time(void) { kmp_kernel_time_acc = 0.0; }
double get_kmp_kernel_time(void) { return kmp_kernel_time_acc; }

void CPF(char pattern[PATTERN_SIZE], int32_t kmpNext[PATTERN_SIZE]) {
  int32_t k = 0;
  int32_t q;
  kmpNext[0] = 0;

c1:
  for (q = 1; q < PATTERN_SIZE; q++) {
c2:
    while (k > 0 && pattern[k] != pattern[q]) {
      /* Prefer table look-up over re-reading pattern every time */
      k = kmpNext[k - 1];
    }
    if (pattern[k] == pattern[q]) {
      ++k;
    }
    kmpNext[q] = k;
  }
}

int kmp(char pattern[PATTERN_SIZE],
        char input[STRING_SIZE],
        int32_t kmpNext[PATTERN_SIZE],
        int32_t n_matches[1]) {
  int32_t i;
  int32_t q;
  n_matches[0] = 0;
  struct timespec kernel_start, kernel_end;

  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  /* Precompute failure function (serial, very small cost for PATTERN_SIZE=4) */
  CPF(pattern, kmpNext);

  q = 0;

#ifdef _OPENMP
  /*
   * Parallelization strategy:
   * - Partition input into contiguous chunks per thread.
   * - Each thread runs a local KMP over its chunk, seeded by the correct
   *   automaton state at the chunk start, and produces its partial match
   *   count.
   * - We perform a cheap prefix pass to propagate the starting state for each
   *   chunk, assuming each chunk begins at a thread-chosen boundary, and then
   *   a parallel-for with reduction to count matches.
   *
   * Since PATTERN_SIZE is tiny (4), state propagation is inexpensive and
   * parallel work over STRING_SIZE dominates.
   */
  {
    int nthreads = 1;
#pragma omp parallel
    {
#pragma omp single
      { nthreads = omp_get_num_threads(); }
    }

    /* Limit number of chunks to STRING_SIZE to avoid zero-length segments */
    if (nthreads > STRING_SIZE)
      nthreads = STRING_SIZE;

    /* Precompute starting index and length for each chunk */
    int32_t chunk_starts[STRING_SIZE];
    int32_t chunk_ends[STRING_SIZE];

    {
      const int32_t base = STRING_SIZE / nthreads;
      const int32_t rem = STRING_SIZE % nthreads;
      int32_t offset = 0;
      int t;
      for (t = 0; t < nthreads; ++t) {
        int32_t len = base + (t < rem ? 1 : 0);
        chunk_starts[t] = offset;
        chunk_ends[t] = offset + len;
        offset += len;
      }
    }

    /* Serial prefix pass to compute initial automaton state per chunk */
    int32_t init_state[STRING_SIZE];
    {
      int32_t state = 0;
      int t;
      init_state[0] = 0;
      for (t = 0; t < nthreads - 1; ++t) {
        int32_t start = chunk_starts[t];
        int32_t end = chunk_ends[t];
        for (i = start; i < end; ++i) {
          while (state > 0 && pattern[state] != input[i]) {
            state = kmpNext[state - 1];
          }
          if (pattern[state] == input[i]) {
            ++state;
          }
          if (state >= PATTERN_SIZE) {
            state = kmpNext[state - 1];
          }
        }
        init_state[t + 1] = state;
      }
    }

    int32_t total_matches = 0;

#pragma omp parallel for reduction(+ : total_matches) schedule(static)
    for (int t = 0; t < nthreads; ++t) {
      int32_t local_q = init_state[t];
      int32_t local_matches = 0;
      int32_t start = chunk_starts[t];
      int32_t end = chunk_ends[t];

      for (i = start; i < end; ++i) {
        while (local_q > 0 && pattern[local_q] != input[i]) {
          local_q = kmpNext[local_q - 1];
        }
        if (pattern[local_q] == input[i]) {
          ++local_q;
        }
        if (local_q >= PATTERN_SIZE) {
          ++local_matches;
          local_q = kmpNext[local_q - 1];
        }
      }
      total_matches += local_matches;
    }

    n_matches[0] = total_matches;
  }
#else
  /* Baseline serial KMP over entire input */
k1:
  for (i = 0; i < STRING_SIZE; i++) {
  k2:
    while (q > 0 && pattern[q] != input[i]) {
      q = kmpNext[q - 1];
    }
    if (pattern[q] == input[i]) {
      ++q;
    }
    if (q >= PATTERN_SIZE) {
      ++n_matches[0];
      q = kmpNext[q - 1];
    }
  }
#endif

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  kmp_kernel_time_acc +=
      (kernel_end.tv_sec - kernel_start.tv_sec) +
      (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
  return 0;
}
