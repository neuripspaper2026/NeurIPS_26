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

#pragma omp simd reduction(+:k)
    for (int32_t q = 1; q < PATTERN_SIZE; q++) {
        while (k > 0 && pattern[k] != pattern[q]) {
            k = kmpNext[k];
        }
        if (pattern[k] == pattern[q]) {
            ++k;
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
    int32_t local_matches = 0;

#ifdef _OPENMP
#pragma omp parallel
    {
        int32_t private_q = 0;
        int32_t private_matches = 0;

#pragma omp for nowait
        for (i = 0; i < STRING_SIZE; i++) {
            while (private_q > 0 && pattern[private_q] != input[i]) {
                private_q = kmpNext[private_q];
            }
            if (pattern[private_q] == input[i]) {
                ++private_q;
            }
            if (private_q >= PATTERN_SIZE) {
                ++private_matches;
                private_q = kmpNext[private_q - 1];
            }
        }

#pragma omp atomic
        local_matches += private_matches;
    }
#else
    for (i = 0; i < STRING_SIZE; i++) {
        while (q > 0 && pattern[q] != input[i]) {
            q = kmpNext[q];
        }
        if (pattern[q] == input[i]) {
            ++q;
        }
        if (q >= PATTERN_SIZE) {
            ++local_matches;
            q = kmpNext[q - 1];
        }
    }
#endif

    n_matches[0] = local_matches;

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    kmp_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                           (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
    return 0;
}
