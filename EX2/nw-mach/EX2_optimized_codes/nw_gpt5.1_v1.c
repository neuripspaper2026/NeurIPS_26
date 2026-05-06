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

    /* Initialize first row */
    {
        int a;
        for (a = 0; a < (ALEN+1); a++) {
            M[a] = a * GAP_SCORE;
        }
    }

    /* Initialize first column */
    {
        int b;
        int stride = (ALEN + 1);
        for (b = 0; b < (BLEN+1); b++) {
            M[b * stride] = b * GAP_SCORE;
        }
    }

    /* Matrix filling loop: wavefront parallelization over anti-diagonals */
    {
        const int stride = (ALEN + 1);
        const int max_k = ALEN + BLEN;

        for (int k = 2; k <= max_k; ++k) {
            /* b runs so that:
               1 <= b <= BLEN
               1 <= a = k - b <= ALEN  ->  1 <= k - b <= ALEN
               -> b in [max(1, k-ALEN), min(BLEN, k-1)]
            */
            int b_start = k - ALEN;
            if (b_start < 1) b_start = 1;
            int b_end = k - 1;
            if (b_end > BLEN) b_end = BLEN;

            if (b_start <= b_end) {
#ifdef _OPENMP
#pragma omp parallel for private(score, up_left, up, left, max, row, row_up) schedule(static)
#endif
                for (int b = b_start; b <= b_end; ++b) {
                    int a = k - b;

                    /* character comparison */
                    score = (SEQA[a - 1] == SEQB[b - 1]) ? MATCH_SCORE : MISMATCH_SCORE;

                    row_up = (b - 1) * stride;
                    row    =  b      * stride;

                    up_left = M[row_up + (a - 1)] + score;
                    up      = M[row_up +  a      ] + GAP_SCORE;
                    left    = M[row    + (a - 1)] + GAP_SCORE;

                    max = MAX(up_left, MAX(up, left));

                    M[row + a] = max;
                    if (max == left) {
                        ptr[row + a] = SKIPB;
                    } else if (max == up) {
                        ptr[row + a] = SKIPA;
                    } else {
                        ptr[row + a] = ALIGN;
                    }
                }
            }
        }
    }

    /* TraceBack (aligned sequences are backwards to avoid string appending) */
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

    /* Pad the result */
    for( ; a_str_idx<ALEN+BLEN; a_str_idx++ ) {
      alignedA[a_str_idx] = '_';
    }
    for( ; b_str_idx<ALEN+BLEN; b_str_idx++ ) {
      alignedB[b_str_idx] = '_';
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    nw_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                          (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
