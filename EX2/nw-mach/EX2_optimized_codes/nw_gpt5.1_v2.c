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

    // Initialize first row
    init_row:
    for(a_idx = 0; a_idx <= ALEN; a_idx++){
        M[a_idx] = a_idx * GAP_SCORE;
    }

    // Initialize first column
    init_col:
    for(b_idx = 0; b_idx <= BLEN; b_idx++){
        M[b_idx*(ALEN+1)] = b_idx * GAP_SCORE;
    }

    // Matrix filling loop using wavefront (anti-diagonal) parallelization
    {
        const int n_rows = BLEN + 1;
        const int n_cols = ALEN + 1;
        const int last_diag = n_rows + n_cols - 2; // index of last anti-diagonal

        // d is the sum index: d = row + col; we skip first row/col (d = 0,1)
        fill_out:
        for (int d = 2; d <= last_diag; d++) {

            // Compute parallelizable range on this anti-diagonal
            int row_start = (d - (n_cols - 1));
            if (row_start < 1) row_start = 1;
            int row_end = d;
            if (row_end > (n_rows - 1)) row_end = n_rows - 1;

            #ifdef _OPENMP
            #pragma omp parallel for private(score, up_left, up, left, max, row, row_up) schedule(static)
            #endif
            for (int row_local = row_start; row_local <= row_end; row_local++) {
                int col_local = d - row_local;
                if (col_local < 1 || col_local > ALEN)
                    continue;

                int idx_a = col_local - 1;
                int idx_b = row_local - 1;

                score = (SEQA[idx_a] == SEQB[idx_b]) ? MATCH_SCORE : MISMATCH_SCORE;

                row_up = (row_local - 1) * (ALEN + 1);
                row    =  row_local      * (ALEN + 1);

                int base_up_left = row_up + (col_local - 1);
                int base_up      = row_up +  col_local;
                int base_left    = row    + (col_local - 1);
                int base_cur     = row    +  col_local;

                up_left = M[base_up_left] + score;
                up      = M[base_up]      + GAP_SCORE;
                left    = M[base_left]    + GAP_SCORE;

                max = MAX(up_left, MAX(up, left));

                M[base_cur] = max;
                if(max == left){
                    ptr[base_cur] = SKIPB;
                } else if(max == up){
                    ptr[base_cur] = SKIPA;
                } else{
                    ptr[base_cur] = ALIGN;
                }
            }
        }
    }

    // TraceBack (n.b. aligned sequences are backwards to avoid string appending)
    a_idx = ALEN;
    b_idx = BLEN;
    a_str_idx = 0;
    b_str_idx = 0;

    trace:
    while(a_idx>0 || b_idx>0) {
        r = b_idx*(ALEN+1);
        char p = ptr[r + a_idx];
        if (p == ALIGN){
            alignedA[a_str_idx++] = SEQA[a_idx-1];
            alignedB[b_str_idx++] = SEQB[b_idx-1];
            a_idx--;
            b_idx--;
        }
        else if (p == SKIPB){
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
    pad_a:
    for( ; a_str_idx<ALEN+BLEN; a_str_idx++ ) {
      alignedA[a_str_idx] = '_';
    }
    pad_b:
    for( ; b_str_idx<ALEN+BLEN; b_str_idx++ ) {
      alignedB[b_str_idx] = '_';
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    nw_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                          (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
