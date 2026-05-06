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

    // Preload pattern into registers/cache
    char pattern_local[PATTERN_SIZE];
    int32_t kmpNext_local[PATTERN_SIZE];
    for(int p = 0; p < PATTERN_SIZE; p++) {
        pattern_local[p] = pattern[p];
        kmpNext_local[p] = kmpNext[p];
    }

#ifdef _OPENMP
    // Parallel search with chunked approach
    const int32_t chunk_size = 4096;
    const int32_t num_chunks = (STRING_SIZE + chunk_size - 1) / chunk_size;
    int32_t total_matches = 0;
    
    #pragma omp parallel
    {
        int32_t local_matches = 0;
        
        #pragma omp for schedule(dynamic, 1) nowait
        for(int32_t chunk = 0; chunk < num_chunks; chunk++) {
            int32_t start = chunk * chunk_size;
            int32_t end = start + chunk_size;
            if(end > STRING_SIZE) end = STRING_SIZE;
            
            // Each chunk needs to maintain its own state
            int32_t q_local = 0;
            
            // Handle overlap from previous chunk by backing up PATTERN_SIZE-1
            int32_t actual_start = (start > PATTERN_SIZE - 1) ? (start - PATTERN_SIZE + 1) : 0;
            
            for(i = actual_start; i < end; i++){
                while (q_local > 0 && pattern_local[q_local] != input[i]){
                    q_local = kmpNext_local[q_local];
                }
                if (pattern_local[q_local] == input[i]){
                    q_local++;
                }
                if (q_local >= PATTERN_SIZE){
                    // Only count matches in the actual chunk range
                    if(i >= start) {
                        local_matches++;
                    }
                    q_local = kmpNext_local[q_local - 1];
                }
            }
        }
        
        #pragma omp atomic
        total_matches += local_matches;
    }
    
    n_matches[0] = total_matches;
#else
    // Serial optimized version
    q = 0;
    
    // Manual loop unrolling for better ILP
    int32_t i_end = STRING_SIZE - (STRING_SIZE % 4);
    
    for(i = 0; i < i_end; i += 4){
        // Iteration 0
        while (q > 0 && pattern_local[q] != input[i]){
            q = kmpNext_local[q];
        }
        if (pattern_local[q] == input[i]){
            q++;
        }
        if (q >= PATTERN_SIZE){
            n_matches[0]++;
            q = kmpNext_local[q - 1];
        }
        
        // Iteration 1
        while (q > 0 && pattern_local[q] != input[i+1]){
            q = kmpNext_local[q];
        }
        if (pattern_local[q] == input[i+1]){
            q++;
        }
        if (q >= PATTERN_SIZE){
            n_matches[0]++;
            q = kmpNext_local[q - 1];
        }
        
        // Iteration 2
        while (q > 0 && pattern_local[q] != input[i+2]){
            q = kmpNext_local[q];
        }
        if (pattern_local[q] == input[i+2]){
            q++;
        }
        if (q >= PATTERN_SIZE){
            n_matches[0]++;
            q = kmpNext_local[q - 1];
        }
        
        // Iteration 3
        while (q > 0 && pattern_local[q] != input[i+3]){
            q = kmpNext_local[q];
        }
        if (pattern_local[q] == input[i+3]){
            q++;
        }
        if (q >= PATTERN_SIZE){
            n_matches[0]++;
            q = kmpNext_local[q - 1];
        }
    }
    
    // Handle remaining iterations
    for(i = i_end; i < STRING_SIZE; i++){
        while (q > 0 && pattern_local[q] != input[i]){
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
