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
        /* Initialize first row */
        int *M_row0 = M;
        for (a_idx = 0; a_idx < (ALEN + 1); ++a_idx) {
            M_row0[a_idx] = a_idx * GAP_SCORE;
        }

        /* Initialize first column */
        int stride = ALEN + 1;
        int *M_col = M;
        for (b_idx = 0; b_idx < (BLEN + 1); ++b_idx) {
            *M_col = b_idx * GAP_SCORE;
            M_col += stride;
        }
    }

    /* Matrix filling loop */
    {
        const int stride = ALEN + 1;
        for (b_idx = 1; b_idx < (BLEN + 1); ++b_idx) {
            row_up = (b_idx - 1) * stride;
            row    =  b_idx      * stride;
            const char *seqA = SEQA;
            const char *seqB = SEQB + (b_idx - 1);
            int *M_row    = M + row;
            int *M_row_up = M + row_up;
            char *ptr_row = ptr + row;

            for (a_idx = 1; a_idx < (ALEN + 1); ++a_idx) {
                score   = (seqA[a_idx - 1] == *seqB) ? MATCH_SCORE : MISMATCH_SCORE;
                up_left = M_row_up[a_idx - 1] + score;
                up      = M_row_up[a_idx]     + GAP_SCORE;
                left    = M_row[a_idx - 1]    + GAP_SCORE;

                max = up_left;
                char dir = ALIGN;
                if (up > max) {
                    max = up;
                    dir = SKIPA;
                }
                if (left > max) {
                    max = left;
                    dir = SKIPB;
                }

                M_row[a_idx]  = max;
                ptr_row[a_idx] = dir;
            }
        }
    }

    /* TraceBack (aligned sequences are backwards to avoid string appending) */
    a_idx = ALEN;
    b_idx = BLEN;
    a_str_idx = 0;
    b_str_idx = 0;

    while (a_idx > 0 || b_idx > 0) {
        r = b_idx * (ALEN + 1);
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
        } else { /* SKIPA */
            alignedA[a_str_idx++] = '-';
            alignedB[b_str_idx++] = SEQB[b_idx - 1];
            --b_idx;
        }
    }

    /* Pad the result */
    while (a_str_idx < ALEN + BLEN) {
        alignedA[a_str_idx++] = '_';
    }
    while (b_str_idx < ALEN + BLEN) {
        alignedB[b_str_idx++] = '_';
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    nw_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                          (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
