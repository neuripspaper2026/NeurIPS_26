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
    init_row:
    for (a_idx = 0; a_idx < (ALEN + 1); a_idx++) {
#pragma HLS UNROLL factor=4
        M[a_idx] = a_idx * GAP_SCORE;
    }

    /* Initialize first column */
    init_col:
    for (b_idx = 0; b_idx < (BLEN + 1); b_idx++) {
#pragma HLS UNROLL factor=4
        M[b_idx * (ALEN + 1)] = b_idx * GAP_SCORE;
    }

    /* Matrix filling loop:
     * Standard DP dependency along anti-diagonals; OpenMP parallelized
     * over cells in each anti-diagonal.
     */
    {
        const int nrow = BLEN + 1;
        const int ncol = ALEN + 1;
        const int max_k = nrow + ncol - 2; /* last diagonal index */

        /* k is the sum index: k = b_idx + a_idx, with b_idx>=1, a_idx>=1 */
        for (int k = 2; k <= max_k; ++k) {
            int b_start = (k - (ncol - 1));
            if (b_start < 1) b_start = 1;
            int b_end = (k - 1);
            if (b_end > (nrow - 1)) b_end = nrow - 1;

#ifdef _OPENMP
#pragma omp parallel for private(score, up_left, up, left, max, row, row_up) schedule(static)
#endif
            for (int b = b_start; b <= b_end; ++b) {
                int a = k - b;
                /* Bounds ensure 1 <= a <= ALEN, 1 <= b <= BLEN */
                if (SEQA[a - 1] == SEQB[b - 1]) {
                    score = MATCH_SCORE;
                } else {
                    score = MISMATCH_SCORE;
                }

                row_up = (b - 1) * (ALEN + 1);
                row    = b * (ALEN + 1);

                up_left = M[row_up + (a - 1)] + score;
                up      = M[row_up + a] + GAP_SCORE;
                left    = M[row + (a - 1)] + GAP_SCORE;

                /* Manual max to avoid macro nesting overhead */
                max = up_left;
                if (up > max)   max = up;
                if (left > max) max = left;

                M[row + a] = max;

                char dir;
                if (max == left) {
                    dir = SKIPB;
                } else if (max == up) {
                    dir = SKIPA;
                } else {
                    dir = ALIGN;
                }
                ptr[row + a] = dir;
            }
        }
    }

    /* TraceBack (aligned sequences are backwards to avoid string appending) */
    a_idx = ALEN;
    b_idx = BLEN;
    a_str_idx = 0;
    b_str_idx = 0;

    trace:
    while (a_idx > 0 || b_idx > 0) {
        r = b_idx * (ALEN + 1);
        char dir = ptr[r + a_idx];
        if (dir == ALIGN) {
            alignedA[a_str_idx++] = SEQA[a_idx - 1];
            alignedB[b_str_idx++] = SEQB[b_idx - 1];
            a_idx--;
            b_idx--;
        } else if (dir == SKIPB) {
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
    pad_a:
    for ( ; a_str_idx < ALEN + BLEN; a_str_idx++ ) {
        alignedA[a_str_idx] = '_';
    }
    pad_b:
    for ( ; b_str_idx < ALEN + BLEN; b_str_idx++ ) {
        alignedB[b_str_idx] = '_';
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    nw_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                          (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
