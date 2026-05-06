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

    int score;
    int up_left, up, left, max;
    int row, row_up;
    int a_idx, b_idx;
    int a_str_idx, b_str_idx;
    int r;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    // Initialize first row
    for(a_idx = 0; a_idx <= ALEN; a_idx++){
        M[a_idx] = a_idx * GAP_SCORE;
    }

    // Initialize first column
    row = 0;
    for(b_idx = 0; b_idx <= BLEN; b_idx++){
        M[row] = b_idx * GAP_SCORE;
        row += (ALEN + 1);
    }

    // Matrix filling loop with OpenMP parallelization on inner dimension
    for(b_idx = 1; b_idx <= BLEN; b_idx++){
        row_up = (b_idx - 1) * (ALEN + 1);
        row    =  b_idx      * (ALEN + 1);
        int base_b = b_idx - 1;
        int local_row_up = row_up;
        int local_row    = row;
#ifdef _OPENMP
#pragma omp parallel for private(a_idx, score, up_left, up, left, max) schedule(static)
#endif
        for(a_idx = 1; a_idx <= ALEN; a_idx++){
            int base_a = a_idx - 1;

            // Use direct index arithmetic with cached row offsets
            if(SEQA[base_a] == SEQB[base_b]){
                score = MATCH_SCORE;
            } else {
                score = MISMATCH_SCORE;
            }

            int idx_left_up = local_row_up + base_a;
            int idx_up      = local_row_up + a_idx;
            int idx_left    = local_row    + base_a;
            int idx         = local_row    + a_idx;

            up_left = M[idx_left_up] + score;
            up      = M[idx_up]      + GAP_SCORE;
            left    = M[idx_left]    + GAP_SCORE;

            // Inline MAX logic to avoid macro expansion overhead
            max = up_left;
            if(up > max)   max = up;
            if(left > max) max = left;

            M[idx] = max;

            // Branch ordering corresponds to most likely cases
            if(max == left){
                ptr[idx] = SKIPB;
            } else if(max == up){
                ptr[idx] = SKIPA;
            } else {
                ptr[idx] = ALIGN;
            }
        }
    }

    // TraceBack (aligned sequences are backwards to avoid string appending)
    a_idx = ALEN;
    b_idx = BLEN;
    a_str_idx = 0;
    b_str_idx = 0;

    while(a_idx > 0 || b_idx > 0) {
        r = b_idx * (ALEN + 1);
        char dir = ptr[r + a_idx];

        if (dir == ALIGN){
            alignedA[a_str_idx++] = SEQA[a_idx-1];
            alignedB[b_str_idx++] = SEQB[b_idx-1];
            a_idx--;
            b_idx--;
        }
        else if (dir == SKIPB){
            alignedA[a_str_idx++] = SEQA[a_idx-1];
            alignedB[b_str_idx++] = '-';
            a_idx--;
        }
        else { // SKIPA
            alignedA[a_str_idx++] = '-';
            alignedB[b_str_idx++] = SEQB[b_idx-1];
            b_idx--;
        }
    }

    // Pad the result
    for( ; a_str_idx < ALEN+BLEN; a_str_idx++ ) {
      alignedA[a_str_idx] = '_';
    }
    for( ; b_str_idx < ALEN+BLEN; b_str_idx++ ) {
      alignedB[b_str_idx] = '_';
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    nw_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                          (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
