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

    // Preload pattern and kmpNext into registers/cache
    char pattern_local[PATTERN_SIZE];
    int32_t kmpNext_local[PATTERN_SIZE];
    for(int32_t p = 0; p < PATTERN_SIZE; p++) {
        pattern_local[p] = pattern[p];
        kmpNext_local[p] = kmpNext[p];
    }

#ifdef _OPENMP
    // Parallel KMP search with chunked approach
    const int32_t chunk_size = 1024;
    const int32_t num_chunks = (STRING_SIZE + chunk_size - 1) / chunk_size;
    int32_t local_matches = 0;

    #pragma omp parallel reduction(+:local_matches)
    {
        int32_t thread_matches = 0;
        #pragma omp for schedule(dynamic, 1) nowait
        for(int32_t chunk = 0; chunk < num_chunks; chunk++) {
            int32_t start = chunk * chunk_size;
            int32_t end = start + chunk_size;
            if(end > STRING_SIZE) end = STRING_SIZE;
            
            // Each chunk starts fresh with q=0
            int32_t q_local = 0;
            
            for(int32_t i = start; i < end; i++) {
                char input_char = input[i];
                
                while(q_local > 0 && pattern_local[q_local] != input_char) {
                    q_local = kmpNext_local[q_local];
                }
                
                if(pattern_local[q_local] == input_char) {
                    q_local++;
                }
                
                if(q_local >= PATTERN_SIZE) {
                    thread_matches++;
                    q_local = kmpNext_local[PATTERN_SIZE - 1];
                }
            }
        }
        local_matches += thread_matches;
    }
    n_matches[0] = local_matches;
#else
    // Serial optimized version
    q = 0;
    k1 : for(i = 0; i < STRING_SIZE; i++){
        char input_char = input[i];
        
        k2 : while (q > 0 && pattern_local[q] != input_char){
            q = kmpNext_local[q];
        }
        if (pattern_local[q] == input_char){
            q++;
        }
        if (q >= PATTERN_SIZE){
            n_matches[0]++;
            q = kmpNext_local[PATTERN_SIZE - 1];
        }
    }
#endif

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    kmp_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                           (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
    return 0;
}
