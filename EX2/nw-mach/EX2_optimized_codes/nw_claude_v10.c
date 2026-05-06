#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../nw.h"

static double nw_kernel_time_acc = 0.0;

void reset_nw_kernel_time(void) { nw_kernel_time_acc = 0.0; }
double get_nw_kernel_time(void) { return nw_kernel_time_acc; }

#define MATCH_SCORE 1
#define MISMATCH_SCORE -1
#define GAP_SCORE -1

#define ALIGN '\\'
#define SKIPA '^'
#define SKIPB '<'

#define MAX(A,B) ( ((A)>(B))?(A):(B) )

void needwun(char SEQA[ALEN], char SEQB[BLEN],
             char alignedA[ALEN+BLEN], char alignedB[ALEN+BLEN],
             int M[(ALEN+1)*(BLEN+1)], char ptr[(ALEN+1)*(BLEN+1)]){

    int score, up_left, up, left, max;
    int row, row_up, r;
    int a_idx, b_idx;
    int a_str_idx, b_str_idx;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    // Initialize first row and column in parallel
    #pragma omp parallel
    {
        #pragma omp for nowait
        for(int i=0; i<(ALEN+1); i++){
            M[i] = i * GAP_SCORE;
        }
        #pragma omp for nowait
        for(int i=0; i<(BLEN+1); i++){
            M[i*(ALEN+1)] = i * GAP_SCORE;
        }
    }

    // Matrix filling with wavefront/anti-diagonal parallelization
    const int total_diags = ALEN + BLEN - 1;
    
    for(int diag = 0; diag < total_diags; diag++){
        int start_b = (diag < BLEN) ? (diag + 1) : BLEN;
        int start_a = (diag < BLEN) ? 1 : (diag - BLEN + 2);
        int diag_len = start_b - start_a + 1;
        
        if(diag_len <= 0) continue;
        
        #pragma omp parallel for schedule(static) if(diag_len > 16)
        for(int k = 0; k < diag_len; k++){
            int b_idx_local = start_b - k;
            int a_idx_local = start_a + k;
            
            if(a_idx_local > ALEN || b_idx_local < 1) continue;
            
            int score_local;
            if(SEQA[a_idx_local-1] == SEQB[b_idx_local-1]){
                score_local = MATCH_SCORE;
            } else {
                score_local = MISMATCH_SCORE;
            }

            int row_up_local = (b_idx_local-1)*(ALEN+1);
            int row_local = b_idx_local*(ALEN+1);

            int up_left_local = M[row_up_local + (a_idx_local-1)] + score_local;
            int up_local      = M[row_up_local + a_idx_local] + GAP_SCORE;
            int left_local    = M[row_local + (a_idx_local-1)] + GAP_SCORE;

            int max_local = MAX(up_left_local, MAX(up_local, left_local));

            M[row_local + a_idx_local] = max_local;
            
            char ptr_val;
            if(max_local == left_local){
                ptr_val = SKIPB;
            } else if(max_local == up_local){
                ptr_val = SKIPA;
            } else{
                ptr_val = ALIGN;
            }
            ptr[row_local + a_idx_local] = ptr_val;
        }
    }

    // TraceBack (sequential, data-dependent)
    a_idx = ALEN;
    b_idx = BLEN;
    a_str_idx = 0;
    b_str_idx = 0;

    while(a_idx>0 || b_idx>0) {
        r = b_idx*(ALEN+1);
        char ptr_val = ptr[r + a_idx];
        if (ptr_val == ALIGN){
            alignedA[a_str_idx++] = SEQA[a_idx-1];
            alignedB[b_str_idx++] = SEQB[b_idx-1];
            a_idx--;
            b_idx--;
        }
        else if (ptr_val == SKIPB){
            alignedA[a_str_idx++] = SEQA[a_idx-1];
            alignedB[b_str_idx++] = '-';
            a_idx--;
        }
        else{
            alignedA[a_str_idx++] = '-';
            alignedB[b_str_idx++] = SEQB[b_idx-1];
            b_idx--;
        }
    }

    // Pad the result in parallel
    #pragma omp parallel
    {
        #pragma omp for nowait
        for(int i = a_str_idx; i < ALEN+BLEN; i++) {
            alignedA[i] = '_';
        }
        #pragma omp for nowait
        for(int i = b_str_idx; i < ALEN+BLEN; i++) {
            alignedB[i] = '_';
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    nw_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                          (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
