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

    // Serial optimization: preload pattern and kmpNext into registers/cache
    char pattern_local[PATTERN_SIZE];
    int32_t kmpNext_local[PATTERN_SIZE];
    for(int32_t p = 0; p < PATTERN_SIZE; p++) {
        pattern_local[p] = pattern[p];
        kmpNext_local[p] = kmpNext[p];
    }

#ifdef _OPENMP
    // Parallel optimization: divide the input string into chunks
    // Each thread processes a chunk with overlap (PATTERN_SIZE-1) to catch matches at boundaries
    int32_t local_matches = 0;
    
    #pragma omp parallel reduction(+:local_matches)
    {
        int num_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        
        // Calculate chunk boundaries with overlap
        int32_t chunk_size = (STRING_SIZE + num_threads - 1) / num_threads;
        int32_t start = thread_id * chunk_size;
        int32_t end = start + chunk_size + PATTERN_SIZE - 1;
        if(end > STRING_SIZE) end = STRING_SIZE;
        if(start >= STRING_SIZE) start = STRING_SIZE;
        
        // Each thread runs KMP on its chunk
        int32_t local_q = 0;
        for(int32_t local_i = start; local_i < end; local_i++) {
            while (local_q > 0 && pattern_local[local_q] != input[local_i]) {
                local_q = kmpNext_local[local_q];
            }
            if (pattern_local[local_q] == input[local_i]) {
                local_q++;
            }
            if (local_q >= PATTERN_SIZE) {
                // Only count matches that start in our non-overlapping region
                int32_t match_start = local_i - PATTERN_SIZE + 1;
                if(match_start >= start && match_start < start + chunk_size) {
                    local_matches++;
                }
                local_q = kmpNext_local[local_q - 1];
            }
        }
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
