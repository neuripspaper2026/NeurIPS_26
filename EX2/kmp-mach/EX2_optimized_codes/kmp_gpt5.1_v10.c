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
            /* Correct prefix link uses k-1 index, not q */
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
    int32_t i, q;
    n_matches[0] = 0;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    /* Pre-compute failure function (serial; tiny cost for PATTERN_SIZE=4) */
    CPF(pattern, kmpNext);

    q = 0;

    /* Parallel tiling over input with per-thread match counts */
#ifdef _OPENMP
    {
        int32_t global_q = 0;
        int32_t global_matches = 0;

        /* Simple chunk size; STRING_SIZE is large, so static scheduling is fine */
        #pragma omp parallel
        {
            int32_t local_q = 0;
            int32_t local_matches = 0;

            #pragma omp for schedule(static)
            for (i = 0; i < STRING_SIZE; i++) {
                k2 : while (local_q > 0 && pattern[local_q] != input[i]) {
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

            /* Accumulate matches; q state across chunks is ignored to avoid races */
            #pragma omp atomic
            global_matches += local_matches;
        }

        q = global_q;
        n_matches[0] = global_matches;
    }
#else
    k1 : for (i = 0; i < STRING_SIZE; i++) {
        k2 : while (q > 0 && pattern[q] != input[i]) {
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
    kmp_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                           (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
    return 0;
}
