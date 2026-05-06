#include <time.h>
#include "../kmp.h"

static double kmp_kernel_time_acc = 0.0;

void reset_kmp_kernel_time(void) { kmp_kernel_time_acc = 0.0; }
double get_kmp_kernel_time(void) { return kmp_kernel_time_acc; }

void CPF(char pattern[PATTERN_SIZE], int32_t kmpNext[PATTERN_SIZE]) {
    int32_t k = 0;
    int32_t q;

    kmpNext[0] = 0;

    c1 : for (q = 1; q < PATTERN_SIZE; q++) {
        char pq = pattern[q];
        c2 : while (k > 0 && pattern[k] != pq) {
            k = kmpNext[k - 1];
        }
        if (pattern[k] == pq) {
            k++;
        }
        kmpNext[q] = k;
    }
}

int kmp(char pattern[PATTERN_SIZE], char input[STRING_SIZE], int32_t kmpNext[PATTERN_SIZE], int32_t n_matches[1]) {
    int32_t i;
    int32_t q = 0;
    n_matches[0] = 0;

    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    CPF(pattern, kmpNext);

    k1 : for (i = 0; i < STRING_SIZE; i++) {
        char ci = input[i];
        k2 : while (q > 0 && pattern[q] != ci) {
            q = kmpNext[q - 1];
        }
        if (pattern[q] == ci) {
            q++;
        }
        if (q >= PATTERN_SIZE) {
            n_matches[0]++;
            q = kmpNext[q - 1];
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    kmp_kernel_time_acc +=
        (double)(kernel_end.tv_sec - kernel_start.tv_sec) +
        (double)(kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;

    return 0;
}
