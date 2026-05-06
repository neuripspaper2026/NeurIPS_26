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
    int row, row_up;
    int a_idx, b_idx;
    int a_str_idx, b_str_idx;
    int r;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    /* Initialize first row */
    for (a_idx = 0; a_idx <= ALEN; a_idx++) {
        M[a_idx] = a_idx * GAP_SCORE;
    }

    /* Initialize first column */
    for (b_idx = 0; b_idx <= BLEN; b_idx++) {
        M[b_idx * (ALEN + 1)] = b_idx * GAP_SCORE;
    }

    /* Matrix filling loop
     * Anti-diagonal parallelization: cells with same (a_idx + b_idx) are independent.
     * k ranges from 2 to ALEN+BLEN (since a_idx,b_idx start from 1).
     */
    {
        const int a_len = ALEN;
        const int b_len = BLEN;
        const int stride = a_len + 1;

        for (int k = 2; k <= a_len + b_len; ++k) {
            int a_start = (k - b_len > 1) ? (k - b_len) : 1;
            int a_end   = (k - 1 < a_len) ? (k - 1) : a_len;

            if (a_start > a_end) {
                continue;
            }

            #ifdef _OPENMP
            #pragma omp parallel for private(a_idx,b_idx,row,row_up,score) schedule(static)
            #endif
            for (a_idx = a_start; a_idx <= a_end; ++a_idx) {
                b_idx = k - a_idx;
                row_up = (b_idx - 1) * stride;
                row    = b_idx * stride;

                score = (SEQA[a_idx - 1] == SEQB[b_idx - 1]) ? MATCH_SCORE : MISMATCH_SCORE;

                const int up_left = M[row_up + (a_idx - 1)] + score;
                const int up      = M[row_up + a_idx] + GAP_SCORE;
                const int left    = M[row + (a_idx - 1)] + GAP_SCORE;

                const int max1 = (up_left > up) ? up_left : up;
                const int max  = (max1 > left) ? max1 : left;

                M[row + a_idx] = max;

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
    for ( ; a_str_idx < ALEN + BLEN; a_str_idx++ ) {
        alignedA[a_str_idx] = '_';
    }
    for ( ; b_str_idx < ALEN + BLEN; b_str_idx++ ) {
        alignedB[b_str_idx] = '_';
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    nw_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                          (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
