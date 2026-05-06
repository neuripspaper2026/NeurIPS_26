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

    // Serial optimization: preload pattern bytes for better cache locality
    char pattern_local[PATTERN_SIZE];
    for(int32_t p = 0; p < PATTERN_SIZE; p++) {
        pattern_local[p] = pattern[p];
    }
    
    int32_t kmpNext_local[PATTERN_SIZE];
    for(int32_t p = 0; p < PATTERN_SIZE; p++) {
        kmpNext_local[p] = kmpNext[p];
    }

#ifdef _OPENMP
    // Parallel optimization: divide the input string into chunks
    // Each thread processes a chunk with overlap to handle pattern matches at boundaries
    int num_threads = omp_get_max_threads();
    int32_t *local_matches = (int32_t *)calloc(num_threads, sizeof(int32_t));
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        
        // Calculate chunk boundaries with overlap
        int32_t chunk_size = STRING_SIZE / nthreads;
        int32_t start = tid * chunk_size;
        int32_t end = (tid == nthreads - 1) ? STRING_SIZE : (tid + 1) * chunk_size + PATTERN_SIZE - 1;
        if(end > STRING_SIZE) end = STRING_SIZE;
        
        int32_t q_local = 0;
        int32_t matches = 0;
        
        for(i = start; i < end; i++){
            while (q_local > 0 && pattern_local[q_local] != input[i]){
                q_local = kmpNext_local[q_local];
            }
            if (pattern_local[q_local] == input[i]){
                q_local++;
            }
            if (q_local >= PATTERN_SIZE){
                // Only count matches that start within this thread's non-overlap region
                if(i - PATTERN_SIZE + 1 >= start && i - PATTERN_SIZE + 1 < start + chunk_size) {
                    matches++;
                }
                q_local = kmpNext_local[q_local - 1];
            }
        }
        
        local_matches[tid] = matches;
    }
    
    // Aggregate results
    for(int t = 0; t < num_threads; t++) {
        n_matches[0] += local_matches[t];
    }
    
    free(local_matches);
#else
    // Serial fallback with optimized local copies
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
