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

    int *restrict M_local = M;
    char *restrict ptr_local = ptr;
    char *restrict SEQA_local = SEQA;
    char *restrict SEQB_local = SEQB;
    char *restrict alignedA_local = alignedA;
    char *restrict alignedB_local = alignedB;

    const int width = ALEN + 1;

    for(a_idx = 0; a_idx < width; a_idx++){
        M_local[a_idx] = a_idx * GAP_SCORE;
    }
    for(b_idx = 0; b_idx < (BLEN + 1); b_idx++){
        M_local[b_idx * width] = b_idx * GAP_SCORE;
    }

    for(b_idx = 1; b_idx < (BLEN + 1); b_idx++){
        row_up = (b_idx - 1) * width;
        row    =  b_idx      * width;
        const char sb = SEQB_local[b_idx - 1];
        for(a_idx = 1; a_idx < (ALEN + 1); a_idx++){
            score = (SEQA_local[a_idx - 1] == sb) ? MATCH_SCORE : MISMATCH_SCORE;

            up_left = M_local[row_up + (a_idx - 1)] + score;
            up      = M_local[row_up +  a_idx     ] + GAP_SCORE;
            left    = M_local[row    + (a_idx - 1)] + GAP_SCORE;

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

            M_local[row + a_idx]   = max;
            ptr_local[row + a_idx] = dir;
        }
    }

    a_idx = ALEN;
    b_idx = BLEN;
    a_str_idx = 0;
    b_str_idx = 0;

    while(a_idx > 0 || b_idx > 0) {
        r = b_idx * width;
        char p = ptr_local[r + a_idx];
        if (p == ALIGN){
            alignedA_local[a_str_idx++] = SEQA_local[a_idx - 1];
            alignedB_local[b_str_idx++] = SEQB_local[b_idx - 1];
            a_idx--;
            b_idx--;
        }
        else if (p == SKIPB){
            alignedA_local[a_str_idx++] = SEQA_local[a_idx - 1];
            alignedB_local[b_str_idx++] = '-';
            a_idx--;
        }
        else{ // SKIPA
            alignedA_local[a_str_idx++] = '-';
            alignedB_local[b_str_idx++] = SEQB_local[b_idx - 1];
            b_idx--;
        }
    }

    for( ; a_str_idx < ALEN + BLEN; a_str_idx++ ) {
      alignedA_local[a_str_idx] = '_';
    }
    for( ; b_str_idx < ALEN + BLEN; b_str_idx++ ) {
      alignedB_local[b_str_idx] = '_';
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    nw_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                          (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
