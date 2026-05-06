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

    int score;
    int up_left, up, left, max;
    int row, row_up, r;
    int a_idx, b_idx;
    int a_str_idx, b_str_idx;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    // Initialize first row
    M[0] = 0;
    int *restrict m_row = M;
    for (a_idx = 1; a_idx <= ALEN; ++a_idx) {
        m_row[a_idx] = a_idx * GAP_SCORE;
    }

    // Initialize first column
    int stride = ALEN + 1;
    for (b_idx = 1, r = stride; b_idx <= BLEN; ++b_idx, r += stride) {
        M[r] = b_idx * GAP_SCORE;
    }

    // Matrix filling loop
    for (b_idx = 1; b_idx <= BLEN; ++b_idx) {
        row_up = (b_idx - 1) * stride;
        row    = b_idx * stride;
        const char sb = SEQB[b_idx - 1];
        for (a_idx = 1; a_idx <= ALEN; ++a_idx) {
            score = (SEQA[a_idx - 1] == sb) ? MATCH_SCORE : MISMATCH_SCORE;

            up_left = M[row_up + (a_idx - 1)] + score;
            up      = M[row_up +  a_idx     ] + GAP_SCORE;
            left    = M[row    + (a_idx - 1)] + GAP_SCORE;

            max = up_left;
            char dir = ALIGN;

            if (left > max) {
                max = left;
                dir = SKIPB;
            }
            if (up > max) {
                max = up;
                dir = SKIPA;
            }

            M[row + a_idx]   = max;
            ptr[row + a_idx] = dir;
        }
    }

    // TraceBack (aligned sequences are backwards to avoid string appending)
    a_idx = ALEN;
    b_idx = BLEN;
    a_str_idx = 0;
    b_str_idx = 0;

    while (a_idx > 0 || b_idx > 0) {
        r = b_idx * stride;
        char p = ptr[r + a_idx];
        if (p == ALIGN) {
            alignedA[a_str_idx++] = SEQA[a_idx - 1];
            alignedB[b_str_idx++] = SEQB[b_idx - 1];
            --a_idx;
            --b_idx;
        } else if (p == SKIPB) {
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
