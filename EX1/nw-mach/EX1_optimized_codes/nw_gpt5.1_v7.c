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

    {
        int gap_mul = GAP_SCORE;
        int *M_local = M;
        int limit_a = ALEN + 1;
        int limit_b = BLEN + 1;

        /* Initialize first row */
        init_row:
        for (a_idx = 0; a_idx < limit_a; ++a_idx) {
            M_local[a_idx] = a_idx * gap_mul;
        }

        /* Initialize first column */
        init_col:
        for (b_idx = 0; b_idx < limit_b; ++b_idx) {
            M_local[b_idx * limit_a] = b_idx * gap_mul;
        }

        /* Matrix filling loop */
        fill_out:
        for (b_idx = 1; b_idx < limit_b; ++b_idx) {
            const int b_off      = b_idx * limit_a;
            const int bminus_off = (b_idx - 1) * limit_a;
            const char sb        = SEQB[b_idx - 1];

            fill_in:
            for (a_idx = 1; a_idx < limit_a; ++a_idx) {
                const int idx_cur  = b_off + a_idx;
                const int idx_ul   = bminus_off + (a_idx - 1);
                const int idx_up   = bminus_off + a_idx;
                const int idx_left = b_off + (a_idx - 1);

                score = (SEQA[a_idx - 1] == sb) ? MATCH_SCORE : MISMATCH_SCORE;

                up_left = M_local[idx_ul] + score;
                up      = M_local[idx_up] + gap_mul;
                left    = M_local[idx_left] + gap_mul;

                /* Inline max to avoid macro re-evaluation and function call */
                max = up_left;
                if (up > max)   max = up;
                if (left > max) max = left;

                M_local[idx_cur] = max;

                /* Branchless-ish pointer selection with ordered checks */
                if (max == left) {
                    ptr[idx_cur] = SKIPB;
                } else if (max == up) {
                    ptr[idx_cur] = SKIPA;
                } else {
                    ptr[idx_cur] = ALIGN;
                }
            }
        }
    }

    /* TraceBack (aligned sequences are backwards to avoid string appending) */
    a_idx = ALEN;
    b_idx = BLEN;
    a_str_idx = 0;
    b_str_idx = 0;

    {
        const int stride = ALEN + 1;
        trace:
        while (a_idx > 0 || b_idx > 0) {
            r = b_idx * stride;
            const int idx = r + a_idx;
            const char dir = ptr[idx];

            if (dir == ALIGN) {
                alignedA[a_str_idx++] = SEQA[a_idx - 1];
                alignedB[b_str_idx++] = SEQB[b_idx - 1];
                --a_idx;
                --b_idx;
            } else if (dir == SKIPB) {
                alignedA[a_str_idx++] = SEQA[a_idx - 1];
                alignedB[b_str_idx++] = '-';
                --a_idx;
            } else { /* SKIPA */
                alignedA[a_str_idx++] = '-';
                alignedB[b_str_idx++] = SEQB[b_idx - 1];
                --b_idx;
            }
        }
    }

    /* Pad the result */
    pad_a:
    for ( ; a_str_idx < ALEN + BLEN; ++a_str_idx ) {
        alignedA[a_str_idx] = '_';
    }
    pad_b:
    for ( ; b_str_idx < ALEN + BLEN; ++b_str_idx ) {
        alignedB[b_str_idx] = '_';
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    nw_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                          (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
