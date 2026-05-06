#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../kmp.h"

static double kmp_kernel_time_acc = 0.0;

void reset_kmp_kernel_time(void) { kmp_kernel_time_acc = 0.0; }
double get_kmp_kernel_time(void) { return kmp_kernel_time_acc; }

void CPF(char pattern[PATTERN_SIZE], int32_t kmpNext[PATTERN_SIZE]) {
    int32_t k, q;
    k = 0;
    kmpNext[0] = 0;

    c1 : for(q = 1; q < PATTERN_SIZE; q++){
        c2 : while(k > 0 && pattern[k] != pattern[q]){
            k = kmpNext[q];
        }
        if(pattern[k] == pattern[q]){
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

    // Serial optimizations: cache pattern values, strength reduction
    char p0 = pattern[0];
    char p1 = pattern[1];
    char p2 = pattern[2];
    char p3 = pattern[3];
    
    int32_t kmp0 = kmpNext[0];
    int32_t kmp1 = kmpNext[1];
    int32_t kmp2 = kmpNext[2];
    int32_t kmp3 = kmpNext[3];

    q = 0;
    
    // Main KMP loop with manual unrolling and optimizations
    k1 : for(i = 0; i < STRING_SIZE; i++){
        char curr_input = input[i];
        
        // Inline the inner while loop with explicit state handling
        k2 : while (q > 0) {
            char pattern_q;
            switch(q) {
                case 1: pattern_q = p1; break;
                case 2: pattern_q = p2; break;
                case 3: pattern_q = p3; break;
                default: pattern_q = p0; break;
            }
            
            if (pattern_q == curr_input) break;
            
            switch(q) {
                case 1: q = kmp1; break;
                case 2: q = kmp2; break;
                case 3: q = kmp3; break;
                default: q = kmp0; break;
            }
        }
        
        // Check for match at current state
        char pattern_q;
        switch(q) {
            case 0: pattern_q = p0; break;
            case 1: pattern_q = p1; break;
            case 2: pattern_q = p2; break;
            case 3: pattern_q = p3; break;
            default: pattern_q = p0; break;
        }
        
        if (pattern_q == curr_input){
            q++;
        }
        
        if (q >= PATTERN_SIZE){
            n_matches[0]++;
            q = kmp3;
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    kmp_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                           (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
    return 0;
}
