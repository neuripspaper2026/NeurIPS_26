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

    // Preload pattern and kmpNext into local variables for better cache locality
    char pattern_local[PATTERN_SIZE];
    int32_t kmpNext_local[PATTERN_SIZE];
    for(int32_t idx = 0; idx < PATTERN_SIZE; idx++) {
        pattern_local[idx] = pattern[idx];
        kmpNext_local[idx] = kmpNext[idx];
    }

#ifdef _OPENMP
    // Parallel block-based search with thread-local match counting
    int32_t total_matches = 0;
    
    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        
        // Calculate block boundaries for this thread
        int32_t block_size = STRING_SIZE / num_threads;
        int32_t start = thread_id * block_size;
        int32_t end = (thread_id == num_threads - 1) ? STRING_SIZE : start + block_size;
        
        // Extend the start backwards by PATTERN_SIZE-1 to handle boundary matches
        int32_t search_start = (start > 0) ? (start - PATTERN_SIZE + 1) : 0;
        if (search_start < 0) search_start = 0;
        
        int32_t local_matches = 0;
        int32_t q_local = 0;
        
        // Perform KMP search on this thread's block
        for(i = search_start; i < end; i++){
            while (q_local > 0 && pattern_local[q_local] != input[i]){
                q_local = kmpNext_local[q_local];
            }
            if (pattern_local[q_local] == input[i]){
                q_local++;
            }
            if (q_local >= PATTERN_SIZE){
                // Only count matches that start within this thread's original range
                int32_t match_start = i - PATTERN_SIZE + 1;
                if (match_start >= start) {
                    local_matches++;
                }
                q_local = kmpNext_local[q_local - 1];
            }
        }
        
        #pragma omp atomic
        total_matches += local_matches;
    }
    
    n_matches[0] = total_matches;
#else
    // Serial optimized version with loop unrolling hints
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
