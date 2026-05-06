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

    // Initialize row and column in parallel
    #pragma omp parallel
    {
        #pragma omp for nowait
        for(int idx=0; idx<(ALEN+1); idx++){
            M[idx] = idx * GAP_SCORE;
        }
        #pragma omp for nowait
        for(int idx=0; idx<(BLEN+1); idx++){
            M[idx*(ALEN+1)] = idx * GAP_SCORE;
        }
    }

    // Matrix filling loop - diagonal wavefront parallelization
    int total_diagonals = ALEN + BLEN - 1;
    
    for(int diag=0; diag < total_diagonals; diag++){
        int start_b = (diag < BLEN) ? (diag + 1) : BLEN;
        int start_a = (diag < BLEN) ? 1 : (diag - BLEN + 2);
        int diag_len = start_b - start_a + 1;
        
        if(start_b > BLEN) start_b = BLEN;
        if(start_a > ALEN) break;
        
        int max_b = (diag < ALEN) ? (diag + 1) : ALEN;
        if(max_b > BLEN) max_b = BLEN;
        
        int actual_start_b = (diag >= BLEN) ? BLEN : (diag + 1);
        int actual_start_a = (diag >= BLEN) ? (diag - BLEN + 2) : 1;
        
        diag_len = actual_start_b - actual_start_a + 1;
        if(diag_len <= 0) continue;
        if(actual_start_a > ALEN || actual_start_b > BLEN) continue;
        
        #pragma omp parallel for schedule(static)
        for(int k=0; k < diag_len; k++){
            int b_idx_local = actual_start_b - k;
            int a_idx_local = actual_start_a + k;
            
            if(a_idx_local > ALEN || b_idx_local < 1) continue;
            
            int local_score;
            if(SEQA[a_idx_local-1] == SEQB[b_idx_local-1]){
                local_score = MATCH_SCORE;
            } else {
                local_score = MISMATCH_SCORE;
            }

            int row_up_local = (b_idx_local-1)*(ALEN+1);
            int row_local = (b_idx_local)*(ALEN+1);

            int up_left_local = M[row_up_local + (a_idx_local-1)] + local_score;
            int up_local      = M[row_up_local + (a_idx_local  )] + GAP_SCORE;
            int left_local    = M[row_local    + (a_idx_local-1)] + GAP_SCORE;

            int max_local = MAX(up_left_local, MAX(up_local, left_local));

            M[row_local + a_idx_local] = max_local;
            if(max_local == left_local){
                ptr[row_local + a_idx_local] = SKIPB;
            } else if(max_local == up_local){
                ptr[row_local + a_idx_local] = SKIPA;
            } else{
                ptr[row_local + a_idx_local] = ALIGN;
            }
        }
    }

    // TraceBack (n.b. aligned sequences are backwards to avoid string appending)
    a_idx = ALEN;
    b_idx = BLEN;
    a_str_idx = 0;
    b_str_idx = 0;

    while(a_idx>0 || b_idx>0) {
        r = b_idx*(ALEN+1);
        if (ptr[r + a_idx] == ALIGN){
            alignedA[a_str_idx++] = SEQA[a_idx-1];
            alignedB[b_str_idx++] = SEQB[b_idx-1];
            a_idx--;
            b_idx--;
        }
        else if (ptr[r + a_idx] == SKIPB){
            alignedA[a_str_idx++] = SEQA[a_idx-1];
            alignedB[b_str_idx++] = '-';
            a_idx--;
        }
        else{ // SKIPA
            alignedA[a_str_idx++] = '-';
            alignedB[b_str_idx++] = SEQB[b_idx-1];
            b_idx--;
        }
    }

    // Pad the result in parallel
    #pragma omp parallel
    {
        #pragma omp for nowait
        for(int idx = a_str_idx; idx < ALEN+BLEN; idx++ ) {
            alignedA[idx] = '_';
        }
        #pragma omp for nowait
        for(int idx = b_str_idx; idx < ALEN+BLEN; idx++ ) {
            alignedB[idx] = '_';
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    nw_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                          (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
