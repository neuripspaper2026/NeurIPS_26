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

    // Serial optimization: load pattern into registers/cache
    char pattern_local[PATTERN_SIZE];
    int32_t kmpNext_local[PATTERN_SIZE];
    for(int32_t p = 0; p < PATTERN_SIZE; p++){
        pattern_local[p] = pattern[p];
        kmpNext_local[p] = kmpNext[p];
    }

#ifdef _OPENMP
    // Parallel optimization: chunk-based processing with thread-local state
    const int32_t chunk_size = 1024;
    const int32_t num_chunks = (STRING_SIZE + chunk_size - 1) / chunk_size;
    
    int32_t *chunk_matches = (int32_t*)calloc(num_chunks, sizeof(int32_t));
    int32_t *chunk_q_state = (int32_t*)malloc(num_chunks * sizeof(int32_t));
    
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for(int32_t chunk_id = 0; chunk_id < num_chunks; chunk_id++){
            int32_t start = chunk_id * chunk_size;
            int32_t end = start + chunk_size;
            if(end > STRING_SIZE) end = STRING_SIZE;
            
            int32_t local_q = 0;
            int32_t local_matches = 0;
            
            for(int32_t i = start; i < end; i++){
                char input_char = input[i];
                while(local_q > 0 && pattern_local[local_q] != input_char){
                    local_q = kmpNext_local[local_q];
                }
                if(pattern_local[local_q] == input_char){
                    local_q++;
                }
                if(local_q >= PATTERN_SIZE){
                    local_matches++;
                    local_q = kmpNext_local[local_q - 1];
                }
            }
            
            chunk_matches[chunk_id] = local_matches;
            chunk_q_state[chunk_id] = local_q;
        }
    }
    
    // Sequential merge: reprocess chunk boundaries
    q = 0;
    int32_t total_matches = 0;
    for(int32_t chunk_id = 0; chunk_id < num_chunks; chunk_id++){
        total_matches += chunk_matches[chunk_id];
        
        // Reprocess boundary region
        int32_t boundary_start = (chunk_id + 1) * chunk_size - (PATTERN_SIZE - 1);
        int32_t boundary_end = (chunk_id + 1) * chunk_size + (PATTERN_SIZE - 1);
        if(boundary_start < 0) boundary_start = 0;
        if(boundary_end > STRING_SIZE) boundary_end = STRING_SIZE;
        if(chunk_id < num_chunks - 1 && boundary_start < boundary_end){
            for(int32_t i = boundary_start; i < boundary_end; i++){
                char input_char = input[i];
                while(q > 0 && pattern_local[q] != input_char){
                    q = kmpNext_local[q];
                }
                if(pattern_local[q] == input_char){
                    q++;
                }
                if(q >= PATTERN_SIZE){
                    total_matches++;
                    q = kmpNext_local[q - 1];
                }
            }
        }
        q = chunk_q_state[chunk_id];
    }
    
    n_matches[0] = total_matches;
    free(chunk_matches);
    free(chunk_q_state);
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
            q = kmpNext_local[q - 1];
        }
    }
#endif

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    kmp_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                           (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
    return 0;
}
