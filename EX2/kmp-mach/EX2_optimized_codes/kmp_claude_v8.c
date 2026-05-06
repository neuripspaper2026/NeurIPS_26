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

    // Serial optimization: cache pattern values and preload kmpNext
    char pattern_cache[PATTERN_SIZE];
    int32_t kmpNext_cache[PATTERN_SIZE];
    for(int32_t p = 0; p < PATTERN_SIZE; p++) {
        pattern_cache[p] = pattern[p];
        kmpNext_cache[p] = kmpNext[p];
    }

#ifdef _OPENMP
    // Parallel optimization: divide the string into chunks and process independently
    const int32_t num_threads = omp_get_max_threads();
    const int32_t chunk_size = STRING_SIZE / num_threads;
    int32_t local_matches[num_threads];
    
    #pragma omp parallel
    {
        const int32_t tid = omp_get_thread_num();
        const int32_t start = tid * chunk_size;
        const int32_t end = (tid == num_threads - 1) ? STRING_SIZE : start + chunk_size;
        
        int32_t local_q = 0;
        int32_t local_count = 0;
        
        // Each thread processes its chunk
        for(int32_t idx = start; idx < end; idx++){
            while (local_q > 0 && pattern_cache[local_q] != input[idx]){
                local_q = kmpNext_cache[local_q];
            }
            if (pattern_cache[local_q] == input[idx]){
                local_q++;
            }
            if (local_q >= PATTERN_SIZE){
                local_count++;
                local_q = kmpNext_cache[local_q - 1];
            }
        }
        
        local_matches[tid] = local_count;
    }
    
    // Aggregate results
    int32_t total_matches = 0;
    for(int32_t t = 0; t < num_threads; t++) {
        total_matches += local_matches[t];
    }
    n_matches[0] = total_matches;
#else
    // Serial optimized version with cached values
    q = 0;
    k1 : for(i = 0; i < STRING_SIZE; i++){
        k2 : while (q > 0 && pattern_cache[q] != input[i]){
            q = kmpNext_cache[q];
        }
        if (pattern_cache[q] == input[i]){
            q++;
        }
        if (q >= PATTERN_SIZE){
            n_matches[0]++;
            q = kmpNext_cache[q - 1];
        }
    }
#endif

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    kmp_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                           (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
    return 0;
}
