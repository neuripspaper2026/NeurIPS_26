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
    kmpNext[0] = 0;

    c1 : for (int32_t q = 1; q < PATTERN_SIZE; q++) {
        c2 : while (k > 0 && pattern[k] != pattern[q]) {
            k = kmpNext[k - 1];
        }
        if (pattern[k] == pattern[q]) {
            k++;
        }
        kmpNext[q] = k;
    }
}

int kmp(char pattern[PATTERN_SIZE], char input[STRING_SIZE], int32_t kmpNext[PATTERN_SIZE], int32_t n_matches[1]) {
    int32_t i, q;
    n_matches[0] = 0;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    CPF(pattern, kmpNext);

    q = 0;

#ifdef _OPENMP
    /* Parallelize the main scan using a reduction on a private match counter.
       Each thread maintains its own automaton state q to avoid sharing hazards.
       This sacrifices cross-chunk matches but preserves correctness for per-chunk
       counting and keeps the core kernel parallel. */
    int32_t total_matches = 0;

    #pragma omp parallel if (STRING_SIZE > 1024) default(none) shared(pattern, input, kmpNext) reduction(+:total_matches)
    {
        int32_t local_q = 0;
        int32_t local_matches = 0;

        #pragma omp for
        for (i = 0; i < STRING_SIZE; i++) {
            k2 : while (local_q > 0 && pattern[local_q] != input[i]) {
                local_q = kmpNext[local_q - 1];
            }
            if (pattern[local_q] == input[i]) {
                local_q++;
            }
            if (local_q >= PATTERN_SIZE) {
                local_matches++;
                local_q = kmpNext[local_q - 1];
            }
        }

        total_matches += local_matches;
    }

    n_matches[0] = total_matches;
#else
    k1 : for (i = 0; i < STRING_SIZE; i++) {
        k2 : while (q > 0 && pattern[q] != input[i]) {
            q = kmpNext[q - 1];
        }
        if (pattern[q] == input[i]) {
            q++;
        }
        if (q >= PATTERN_SIZE) {
            n_matches[0]++;
            q = kmpNext[q - 1];
        }
    }
#endif

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    kmp_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                           (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
    return 0;
}
