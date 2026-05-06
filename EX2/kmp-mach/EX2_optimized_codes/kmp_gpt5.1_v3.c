#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../kmp.h"

static double kmp_kernel_time_acc = 0.0;

void reset_kmp_kernel_time(void) { kmp_kernel_time_acc = 0.0; }
double get_kmp_kernel_time(void) { return kmp_kernel_time_acc; }

static inline void CPF(char pattern[PATTERN_SIZE], int32_t kmpNext[PATTERN_SIZE]) {
    int32_t k = 0;
    kmpNext[0] = 0;

    c1 : for (int32_t q = 1; q < PATTERN_SIZE; ++q) {
        c2 : while (k > 0 && pattern[k] != pattern[q]) {
            k = kmpNext[k - 1];
        }
        if (pattern[k] == pattern[q]) {
            ++k;
        }
        kmpNext[q] = k;
    }
}

int kmp(char pattern[PATTERN_SIZE], char input[STRING_SIZE], int32_t kmpNext[PATTERN_SIZE], int32_t n_matches[1]) {
    int32_t total_matches = 0;
    struct timespec kernel_start, kernel_end;

    n_matches[0] = 0;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    CPF(pattern, kmpNext);

    int32_t pat_len = PATTERN_SIZE;
    int32_t str_len = STRING_SIZE;

#ifdef _OPENMP
    #pragma omp parallel
    {
        int32_t local_matches = 0;

        // Each thread maintains its own KMP state on a private segment
        #pragma omp for nowait schedule(static)
        for (int32_t i = 0; i < str_len; ++i) {
            int32_t q = 0;
            char p0 = pattern[0];  // small pattern; keep first char in a register

            for (int32_t j = i; j < str_len; ++j) {
                char c = input[j];

                // Fast path for mismatch without while-loop when q == 0
                if (q == 0) {
                    if (c != p0) {
                        continue;
                    }
                    q = 1;
                } else {
                    while (q > 0 && pattern[q] != c) {
                        q = kmpNext[q - 1];
                    }
                    if (pattern[q] == c) {
                        ++q;
                    }
                }

                if (q >= pat_len) {
                    ++local_matches;
                    q = kmpNext[q - 1];
                }
            }
        }

        #pragma omp atomic
        total_matches += local_matches;
    }
#else
    {
        int32_t q = 0;
        int32_t pat_len_local = pat_len;

        k1 : for (int32_t i = 0; i < str_len; ++i) {
            char c = input[i];

            k2 : while (q > 0 && pattern[q] != c) {
                q = kmpNext[q - 1];
            }
            if (pattern[q] == c) {
                ++q;
            }
            if (q >= pat_len_local) {
                ++total_matches;
                q = kmpNext[q - 1];
            }
        }
    }
#endif

    n_matches[0] = total_matches;

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    kmp_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                           (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
    return 0;
}
