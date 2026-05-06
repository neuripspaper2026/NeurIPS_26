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

    // Serial optimization: load pattern into local array for better cache locality
    char local_pattern[PATTERN_SIZE];
    int32_t local_kmpNext[PATTERN_SIZE];
    for(int32_t p = 0; p < PATTERN_SIZE; p++){
        local_pattern[p] = pattern[p];
        local_kmpNext[p] = kmpNext[p];
    }

#ifdef _OPENMP
    // Parallel optimization: divide the input string into chunks
    // Each thread processes its chunk independently and counts matches
    int32_t num_threads;
    int32_t *thread_matches;
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            num_threads = omp_get_num_threads();
            thread_matches = (int32_t*)calloc(num_threads, sizeof(int32_t));
        }
    }

    #pragma omp parallel
    {
        int32_t tid = omp_get_thread_num();
        int32_t chunk_size = STRING_SIZE / num_threads;
        int32_t start = tid * chunk_size;
        int32_t end = (tid == num_threads - 1) ? STRING_SIZE : start + chunk_size + PATTERN_SIZE - 1;
        if(end > STRING_SIZE) end = STRING_SIZE;
        
        int32_t local_q = 0;
        int32_t local_count = 0;
        
        for(i = start; i < end; i++){
            while (local_q > 0 && local_pattern[local_q] != input[i]){
                local_q = local_kmpNext[local_q];
            }
            if (local_pattern[local_q] == input[i]){
                local_q++;
            }
            if (local_q >= PATTERN_SIZE){
                // Only count matches that start within this thread's original chunk
                if(i - PATTERN_SIZE + 1 < (tid == num_threads - 1 ? STRING_SIZE : start + chunk_size)){
                    local_count++;
                }
                local_q = local_kmpNext[local_q - 1];
            }
        }
        
        thread_matches[tid] = local_count;
    }

    // Aggregate results
    for(int32_t t = 0; t < num_threads; t++){
        n_matches[0] += thread_matches[t];
    }
    free(thread_matches);
#else
    // Serial execution with optimized local copies
    q = 0;
    k1 : for(i = 0; i < STRING_SIZE; i++){
        k2 : while (q > 0 && local_pattern[q] != input[i]){
            q = local_kmpNext[q];
        }
        if (local_pattern[q] == input[i]){
            q++;
        }
        if (q >= PATTERN_SIZE){
            n_matches[0]++;
            q = local_kmpNext[q - 1];
        }
    }
#endif

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    kmp_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                           (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
    return 0;
}
