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
    int row, row_up;
    int a_idx, b_idx;
    int a_str_idx, b_str_idx;
    int r;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    /* Initialize first row (b_idx = 0) */
    for (a_idx = 0; a_idx < (ALEN + 1); a_idx++) {
        M[a_idx] = a_idx * GAP_SCORE;
    }

    /* Initialize first column (a_idx = 0) */
    for (b_idx = 0; b_idx < (BLEN + 1); b_idx++) {
        M[b_idx * (ALEN + 1)] = b_idx * GAP_SCORE;
    }

    /* Matrix filling loop: process along anti-diagonals for parallelism */
#ifdef _OPENMP
    {
        int diag;
        const int a_max = ALEN;
        const int b_max = BLEN;
        const int a_len1 = ALEN + 1;
        const int total_diags = a_max + b_max;

        for (diag = 2; diag <= total_diags; ++diag) {
            int b_start = (diag > a_max + 1) ? (diag - (a_max + 1)) : 1;
            int b_end   = (diag - 1 < b_max) ? (diag - 1) : b_max;
#pragma omp parallel for default(none) private(b_idx,a_idx,row,row_up,score,up_left,up,left,max) shared(SEQA,SEQB,M,ptr,a_len1,diag,b_start,b_end)
            for (b_idx = b_start; b_idx <= b_end; ++b_idx) {
                a_idx = diag - b_idx;

                /* compute row indices once per cell */
                row_up = (b_idx - 1) * a_len1;
                row    =  b_idx      * a_len1;

                /* match/mismatch score */
                score = (SEQA[a_idx - 1] == SEQB[b_idx - 1]) ? MATCH_SCORE : MISMATCH_SCORE;

                up_left = M[row_up + (a_idx - 1)] + score;
                up      = M[row_up +  a_idx      ] + GAP_SCORE;
                left    = M[row    + (a_idx - 1)] + GAP_SCORE;

                /* branchless-like max selection */
                max = up_left;
                if (up > max)   max = up;
                if (left > max) max = left;

                M[row + a_idx] = max;

                /* tie-breaking order matches original logic: left > up > diag */
                if (max == left) {
                    ptr[row + a_idx] = SKIPB;
                } else if (max == up) {
                    ptr[row + a_idx] = SKIPA;
                } else {
                    ptr[row + a_idx] = ALIGN;
                }
            }
        }
    }
#else
    for (b_idx = 1; b_idx < (BLEN + 1); b_idx++) {
        int row_base     = b_idx * (ALEN + 1);
        int row_up_base  = (b_idx - 1) * (ALEN + 1);
        for (a_idx = 1; a_idx < (ALEN + 1); a_idx++) {
            score = (SEQA[a_idx - 1] == SEQB[b_idx - 1]) ? MATCH_SCORE : MISMATCH_SCORE;

            up_left = M[row_up_base + (a_idx - 1)] + score;
            up      = M[row_up_base +  a_idx      ] + GAP_SCORE;
            left    = M[row_base    + (a_idx - 1)] + GAP_SCORE;

            max = up_left;
            if (up > max)   max = up;
            if (left > max) max = left;

            M[row_base + a_idx] = max;
            if (max == left) {
                ptr[row_base + a_idx] = SKIPB;
            } else if (max == up) {
                ptr[row_base + a_idx] = SKIPA;
            } else {
                ptr[row_base + a_idx] = ALIGN;
            }
        }
    }
#endif

    /* TraceBack (aligned sequences are backwards to avoid string appending) */
    a_idx = ALEN;
    b_idx = BLEN;
    a_str_idx = 0;
    b_str_idx = 0;

    while (a_idx > 0 || b_idx > 0) {
        r = b_idx * (ALEN + 1);
        if (ptr[r + a_idx] == ALIGN) {
            alignedA[a_str_idx++] = SEQA[a_idx - 1];
            alignedB[b_str_idx++] = SEQB[b_idx - 1];
            a_idx--;
            b_idx--;
        } else if (ptr[r + a_idx] == SKIPB) {
            alignedA[a_str_idx++] = SEQA[a_idx - 1];
            alignedB[b_str_idx++] = '-';
            a_idx--;
        } else { /* SKIPA */
            alignedA[a_str_idx++] = '-';
            alignedB[b_str_idx++] = SEQB[b_idx - 1];
            b_idx--;
        }
    }

    /* Pad the result */
    for (; a_str_idx < ALEN + BLEN; a_str_idx++) {
        alignedA[a_str_idx] = '_';
    }
    for (; b_str_idx < ALEN + BLEN; b_str_idx++) {
        alignedB[b_str_idx] = '_';
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    nw_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                          (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
