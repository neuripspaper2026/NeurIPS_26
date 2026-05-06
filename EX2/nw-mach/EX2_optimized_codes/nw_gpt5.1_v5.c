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
    int row, row_up, r;
    int a_idx, b_idx;
    int a_str_idx, b_str_idx;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    /* Precompute frequently used constants */
    const int gap_score = GAP_SCORE;
    const int match_score = MATCH_SCORE;
    const int mismatch_score = MISMATCH_SCORE;
    const int alen1 = ALEN + 1;
    const int blen1 = BLEN + 1;
    const int total_cells = alen1 * blen1;

    /* Initialize first row and column */
    init_row: for(a_idx = 0; a_idx < alen1; a_idx++){
        M[a_idx] = a_idx * gap_score;
    }
    init_col: for(b_idx = 0; b_idx < blen1; b_idx++){
        M[b_idx * alen1] = b_idx * gap_score;
    }

    /* Matrix filling loop: process by anti-diagonals for parallelism */
    {
        int max_diag = ALEN + BLEN;
        int *restrict Mp = M;
        char *restrict ptrp = ptr;
        char *restrict seqA = SEQA;
        char *restrict seqB = SEQB;

        int d;
        for (d = 2; d <= max_diag; d++) {
            int imin = d - BLEN;
            if (imin < 1) imin = 1;
            int imax = d - 1;
            if (imax > ALEN) imax = ALEN;

            if (imin > imax) continue;

            int i;
            #ifdef _OPENMP
            #pragma omp parallel for default(none) shared(Mp,ptrp,seqA,seqB,alen1,blen1,gap_score,match_score,mismatch_score,d,imin,imax) private(i,row,row_up,score)
            #endif
            for (i = imin; i <= imax; i++) {
                int j = d - i;
                /* bounds guaranteed by imin/imax */
                int a = i;
                int b = j;

                char ca = seqA[a - 1];
                char cb = seqB[b - 1];

                score = (ca == cb) ? match_score : mismatch_score;

                row_up = (b - 1) * alen1;
                row    = b * alen1;

                int up_left = Mp[row_up + (a - 1)] + score;
                int up      = Mp[row_up +  a     ] + gap_score;
                int left    = Mp[row    + (a - 1)] + gap_score;

                int max_ul_up = (up_left > up) ? up_left : up;
                int max       = (max_ul_up > left) ? max_ul_up : left;

                Mp[row + a] = max;
                if (max == left){
                    ptrp[row + a] = SKIPB;
                } else if (max == up){
                    ptrp[row + a] = SKIPA;
                } else{
                    ptrp[row + a] = ALIGN;
                }
            }
        }
    }

    /* TraceBack (aligned sequences are backwards to avoid string appending) */
    a_idx = ALEN;
    b_idx = BLEN;
    a_str_idx = 0;
    b_str_idx = 0;

    trace: while(a_idx > 0 || b_idx > 0) {
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
        else{ /* SKIPA */
            alignedA[a_str_idx++] = '-';
            alignedB[b_str_idx++] = SEQB[b_idx-1];
            b_idx--;
        }
    }

    /* Pad the result */
    pad_a: for( ; a_str_idx < ALEN+BLEN; a_str_idx++ ) {
      alignedA[a_str_idx] = '_';
    }
    pad_b: for( ; b_str_idx < ALEN+BLEN; b_str_idx++ ) {
      alignedB[b_str_idx] = '_';
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    nw_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                          (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
