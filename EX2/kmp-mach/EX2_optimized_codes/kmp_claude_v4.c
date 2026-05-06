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

    // Serial optimization: preload pattern into registers/cache
    char pattern_local[PATTERN_SIZE];
    int32_t kmpNext_local[PATTERN_SIZE];
    for(int32_t p = 0; p < PATTERN_SIZE; p++) {
        pattern_local[p] = pattern[p];
        kmpNext_local[p] = kmpNext[p];
    }

#ifdef _OPENMP
    // Parallel KMP matching using chunked approach
    const int32_t chunk_size = 1024;
    const int32_t num_chunks = (STRING_SIZE + chunk_size - 1) / chunk_size;
    int32_t local_matches = 0;

    #pragma omp parallel reduction(+:local_matches)
    {
        int32_t thread_matches = 0;
        
        #pragma omp for schedule(dynamic, 1) nowait
        for(int32_t chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
            int32_t start = chunk_id * chunk_size;
            int32_t end = start + chunk_size + PATTERN_SIZE - 1;
            if(end > STRING_SIZE) end = STRING_SIZE;
            
            int32_t local_q = 0;
            
            for(int32_t idx = start; idx < end; idx++) {
                while(local_q > 0 && pattern_local[local_q] != input[idx]) {
                    local_q = kmpNext_local[local_q];
                }
                if(pattern_local[local_q] == input[idx]) {
                    local_q++;
                }
                if(local_q >= PATTERN_SIZE) {
                    // Only count matches that start within the chunk boundary
                    if(idx - PATTERN_SIZE + 1 >= start && idx - PATTERN_SIZE + 1 < start + chunk_size) {
                        thread_matches++;
                    }
                    local_q = kmpNext_local[local_q - 1];
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
        k2 : while (q > 0 && pattern_local[q] != input[i]){
            q = kmpNext_local[q];
        }
        if (pattern_local[q] == input[i]){
            q++;
        }
        if (q >= PATTERN_SIZE){
            n_matches[0]++;
            q = kmpNext_local[q - 1];
        }
    }
#endif

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    kmp_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                           (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
    return 0;
}
