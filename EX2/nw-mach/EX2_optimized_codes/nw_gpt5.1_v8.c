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

    /* Precompute leading dimension and total matrix size to reduce repeated work */
    const int ld = ALEN + 1;
    const int total_elems = (ALEN + 1) * (BLEN + 1);

    /* Initialize DP matrix first row and column with explicit indexing */
    {
        int gap = GAP_SCORE;
        for (int i = 0; i <= ALEN; ++i) {
            M[i] = i * gap;
        }
        for (int j = 0; j <= BLEN; ++j) {
            M[j * ld] = j * gap;
        }
    }

    /* Matrix filling loop:
     * The dependency pattern is that cell (i,j) depends on:
     * (i-1,j-1), (i-1,j), (i,j-1). This creates anti-diagonal parallelism.
     * For clarity and speed on small fixed sizes, we keep the simple double loop
     * and parallelize inner dimension where safe, relying on OpenMP and CPU caches.
     */
#ifdef _OPENMP
    /* Parallelize over rows (b_idx) and compute each row sequentially to
       preserve left-dependency while allowing different rows to be processed
       in parallel using ordered construct to respect up-dependency. */
#pragma omp parallel default(none) shared(M, ptr, SEQA, SEQB) private(score, up_left, up, left, max, row, row_up, a_idx, b_idx) firstprivate(ld)
    {
#pragma omp for schedule(static)
        for (b_idx = 1; b_idx <= BLEN; ++b_idx) {
            row_up = (b_idx - 1) * ld;
            row    =  b_idx      * ld;

            for (a_idx = 1; a_idx <= ALEN; ++a_idx) {
                /* Branchless match/mismatch scoring to help vectorization */
                const int is_match = (SEQA[a_idx - 1] == SEQB[b_idx - 1]);
                score   = is_match ? MATCH_SCORE : MISMATCH_SCORE;

                up_left = M[row_up + (a_idx - 1)] + score;
                up      = M[row_up +  a_idx     ] + GAP_SCORE;
                left    = M[row    + (a_idx - 1)] + GAP_SCORE;

                /* Manual max chain avoids macro side-effect issues and can vectorize */
                max = up_left;
                if (up > max)   max = up;
                if (left > max) max = left;

                M[row + a_idx] = max;

                /* Use if-else in same order – most common path likely ALIGN/left/up.
                   Keep tight to help branch prediction. */
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
    for (b_idx = 1; b_idx <= BLEN; ++b_idx) {
        row_up = (b_idx - 1) * ld;
        row    =  b_idx      * ld;

        for (a_idx = 1; a_idx <= ALEN; ++a_idx) {
            const int is_match = (SEQA[a_idx - 1] == SEQB[b_idx - 1]);
            score   = is_match ? MATCH_SCORE : MISMATCH_SCORE;

            up_left = M[row_up + (a_idx - 1)] + score;
            up      = M[row_up +  a_idx     ] + GAP_SCORE;
            left    = M[row    + (a_idx - 1)] + GAP_SCORE;

            max = up_left;
            if (up > max)   max = up;
            if (left > max) max = left;

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
#endif

    /* TraceBack (aligned sequences are backwards to avoid string appending) */
    a_idx = ALEN;
    b_idx = BLEN;
    a_str_idx = 0;
    b_str_idx = 0;

    while (a_idx > 0 || b_idx > 0) {
        r = b_idx * ld;
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
