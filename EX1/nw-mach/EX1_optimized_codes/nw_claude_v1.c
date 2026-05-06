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

    int a_idx, b_idx;
    int a_str_idx, b_str_idx;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    const int stride = ALEN + 1;
    
    init_row: for(a_idx=0; a_idx<stride; a_idx++){
        M[a_idx] = a_idx * GAP_SCORE;
    }
    init_col: for(b_idx=1; b_idx<(BLEN+1); b_idx++){
        M[b_idx*stride] = b_idx * GAP_SCORE;
    }

    // Matrix filling loop
    fill_out: for(b_idx=1; b_idx<(BLEN+1); b_idx++){
        int row_up = (b_idx-1)*stride;
        int row = b_idx*stride;
        char seqb_char = SEQB[b_idx-1];
        
        fill_in: for(a_idx=1; a_idx<stride; a_idx++){
            int score = (SEQA[a_idx-1] == seqb_char) ? MATCH_SCORE : MISMATCH_SCORE;

            int up_left = M[row_up + (a_idx-1)] + score;
            int up      = M[row_up + a_idx] + GAP_SCORE;
            int left    = M[row + (a_idx-1)] + GAP_SCORE;

            int max = MAX(up_left, MAX(up, left));

            M[row + a_idx] = max;
            char direction;
            if(max == left){
                direction = SKIPB;
            } else if(max == up){
                direction = SKIPA;
            } else{
                direction = ALIGN;
            }
            ptr[row + a_idx] = direction;
        }
    }

    // TraceBack (n.b. aligned sequences are backwards to avoid string appending)
    a_idx = ALEN;
    b_idx = BLEN;
    a_str_idx = 0;
    b_str_idx = 0;

    trace: while(a_idx>0 || b_idx>0) {
        int r = b_idx*stride;
        char direction = ptr[r + a_idx];
        if (direction == ALIGN){
            alignedA[a_str_idx++] = SEQA[a_idx-1];
            alignedB[b_str_idx++] = SEQB[b_idx-1];
            a_idx--;
            b_idx--;
        }
        else if (direction == SKIPB){
            alignedA[a_str_idx++] = SEQA[a_idx-1];
            alignedB[b_str_idx++] = '-';
            a_idx--;
        }
        else{ // SKIPA
            alignedA[a_str_idx++] = '-';
            alignedB[b_str_idx++] = SEQB[b_idx-1];
            b_idx--;
        }
    }

    // Pad the result
    pad_a: for( ; a_str_idx<ALEN+BLEN; a_str_idx++ ) {
      alignedA[a_str_idx] = '_';
    }
    pad_b: for( ; b_str_idx<ALEN+BLEN; b_str_idx++ ) {
      alignedB[b_str_idx] = '_';
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    nw_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                          (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
