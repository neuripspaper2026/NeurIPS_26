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

    // Initialize first row and column
    #pragma omp parallel
    {
        #pragma omp for nowait
        for(a_idx=0; a_idx<(ALEN+1); a_idx++){
            M[a_idx] = a_idx * GAP_SCORE;
        }
        #pragma omp for nowait
        for(b_idx=0; b_idx<(BLEN+1); b_idx++){
            M[b_idx*(ALEN+1)] = b_idx * GAP_SCORE;
        }
    }

    // Matrix filling loop - diagonal wavefront parallelization
    // Process anti-diagonals to expose parallelism
    int total_diags = ALEN + BLEN - 1;
    
    for(int diag = 0; diag < total_diags; diag++){
        int start_b = (diag < BLEN) ? (diag + 1) : BLEN;
        int start_a = (diag < BLEN) ? 1 : (diag - BLEN + 2);
        int diag_len = start_b - ((diag >= ALEN) ? (diag - ALEN + 1) : 0);
        
        #pragma omp parallel for if(diag_len > 8) schedule(static)
        for(int i = 0; i < diag_len; i++){
            b_idx = start_b - i;
            a_idx = start_a + i;
            
            int local_score;
            if(SEQA[a_idx-1] == SEQB[b_idx-1]){
                local_score = MATCH_SCORE;
            } else {
                local_score = MISMATCH_SCORE;
            }

            row_up = (b_idx-1)*(ALEN+1);
            row = (b_idx)*(ALEN+1);

            up_left = M[row_up + (a_idx-1)] + local_score;
            up      = M[row_up + (a_idx  )] + GAP_SCORE;
            left    = M[row    + (a_idx-1)] + GAP_SCORE;

            int local_max = MAX(up_left, MAX(up, left));

            M[row + a_idx] = local_max;
            if(local_max == left){
                ptr[row + a_idx] = SKIPB;
            } else if(local_max == up){
                ptr[row + a_idx] = SKIPA;
            } else{
                ptr[row + a_idx] = ALIGN;
            }
        }
    }

    // TraceBack (n.b. aligned sequences are backwards to avoid string appending)
    a_idx = ALEN;
    b_idx = BLEN;
    a_str_idx = 0;
    b_str_idx = 0;

    trace: while(a_idx>0 || b_idx>0) {
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

    // Pad the result
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
