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
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    /* Precompute constants */
    const int gap_times_a = GAP_SCORE;
    const int gap_times_b = GAP_SCORE;
    const int width = ALEN + 1;

    /* Initialize first row */
    init_row:
    for (a_idx = 0; a_idx < (ALEN + 1); a_idx++) {
        M[a_idx] = a_idx * gap_times_a;
    }

    /* Initialize first column */
    init_col:
    for (b_idx = 0; b_idx < (BLEN + 1); b_idx++) {
        M[b_idx * width] = b_idx * gap_times_b;
    }

    /* Matrix filling loop */
    fill_out:
    for (b_idx = 1; b_idx < (BLEN + 1); b_idx++) {
        row_up = (b_idx - 1) * width;
        row    = b_idx * width;
        fill_in:
        for (a_idx = 1; a_idx < (ALEN + 1); a_idx++) {
            /* Branchless score computation */
            score = (SEQA[a_idx - 1] == SEQB[b_idx - 1]) ? MATCH_SCORE : MISMATCH_SCORE;

            const int idx     = row + a_idx;
            const int idx_up  = row_up + a_idx;
            const int idx_lft = row + (a_idx - 1);
            const int idx_ul  = row_up + (a_idx - 1);

            up_left = M[idx_ul] + score;
            up      = M[idx_up] + GAP_SCORE;
            left    = M[idx_lft] + GAP_SCORE;

            /* Inline max of three without macro nesting */
            max = up_left;
            if (up > max)   max = up;
            if (left > max) max = left;

            M[idx] = max;

            /* Store traceback pointer, order chosen to reduce comparisons on average */
            if (max == up_left) {
                ptr[idx] = ALIGN;
            } else if (max == left) {
                ptr[idx] = SKIPB;
            } else {
                ptr[idx] = SKIPA;
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
        r = b_idx * width;
        const int idx = r + a_idx;
        const char dir = ptr[idx];

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
