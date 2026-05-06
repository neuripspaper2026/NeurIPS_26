#include <time.h>
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
    int gap_score = GAP_SCORE;
    int match_score = MATCH_SCORE;
    int mismatch_score = MISMATCH_SCORE;
    int width = ALEN + 1;
    int total_rows = BLEN + 1;
    int total_cols = ALEN + 1;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    // Initialize first row
    {
        int acc = 0;
        M[0] = 0;
        for (a_idx = 1; a_idx < total_cols; ++a_idx) {
            acc += gap_score;
            M[a_idx] = acc;
        }
    }

    // Initialize first column
    {
        int acc = 0;
        for (b_idx = 1; b_idx < total_rows; ++b_idx) {
            acc += gap_score;
            M[b_idx * width] = acc;
        }
    }

    // Matrix filling loop
    for (b_idx = 1; b_idx < total_rows; ++b_idx) {
        row_up = (b_idx - 1) * width;
        row    = b_idx * width;
        for (a_idx = 1; a_idx < total_cols; ++a_idx) {
            score = (SEQA[a_idx - 1] == SEQB[b_idx - 1]) ? match_score : mismatch_score;

            up_left = M[row_up + (a_idx - 1)] + score;
            up      = M[row_up +  a_idx     ] + gap_score;
            left    = M[row    + (a_idx - 1)] + gap_score;

            max = up_left;
            if (up > max)   max = up;
            if (left > max) max = left;

            M[row + a_idx] = max;

            // Determine traceback direction; order chosen to minimize comparisons
            if (max == up_left) {
                ptr[row + a_idx] = ALIGN;
            } else if (max == left) {
                ptr[row + a_idx] = SKIPB;
            } else {
                ptr[row + a_idx] = SKIPA;
            }
        }
    }

    // TraceBack (aligned sequences are backwards to avoid string appending)
    a_idx = ALEN;
    b_idx = BLEN;
    a_str_idx = 0;
    b_str_idx = 0;

    while (a_idx > 0 || b_idx > 0) {
        r = b_idx * width;
        char dir = ptr[r + a_idx];
        if (dir == ALIGN) {
            alignedA[a_str_idx++] = SEQA[a_idx - 1];
            alignedB[b_str_idx++] = SEQB[b_idx - 1];
            --a_idx;
            --b_idx;
        } else if (dir == SKIPB) {
            alignedA[a_str_idx++] = SEQA[a_idx - 1];
            alignedB[b_str_idx++] = '-';
            --a_idx;
        } else { // SKIPA
            alignedA[a_str_idx++] = '-';
            alignedB[b_str_idx++] = SEQB[b_idx - 1];
            --b_idx;
        }
    }

    // Pad the result
    for ( ; a_str_idx < ALEN + BLEN; ++a_str_idx ) {
      alignedA[a_str_idx] = '_';
    }
    for ( ; b_str_idx < ALEN + BLEN; ++b_str_idx ) {
      alignedB[b_str_idx] = '_';
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    nw_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                          (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
