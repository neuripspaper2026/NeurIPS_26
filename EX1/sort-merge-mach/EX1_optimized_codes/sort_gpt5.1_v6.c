#include <time.h>
#include "../sort.h"

static double sort_merge_kernel_time_acc = 0.0;

void reset_sort_merge_kernel_time(void) { sort_merge_kernel_time_acc = 0.0; }
double get_sort_merge_kernel_time(void) { return sort_merge_kernel_time_acc; }

void merge(TYPE a[SIZE], int start, int m, int stop){
    TYPE temp[SIZE];
    int i, j, k;

    /* Copy left half directly */
    for(i = start; i <= m; i++){
        temp[i] = a[i];
    }

    /* Copy right half in reverse into upper part of temp */
    for(j = m + 1; j <= stop; j++){
        temp[m + 1 + stop - j] = a[j];
    }

    i = start;
    j = stop;

    /* Merge from temp back into a with reduced memory traffic */
    for(k = start; k <= stop; k++){
        TYPE tmp_i = temp[i];
        TYPE tmp_j = temp[j];
        if(tmp_j < tmp_i) {
            a[k] = tmp_j;
            j--;
        } else {
            a[k] = tmp_i;
            i++;
        }
    }
}

void ms_mergesort(TYPE a[SIZE]) {
    int start, stop;
    int i, m, from, mid, to;

    start = 0;
    stop = SIZE;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    /* Bottom-up iterative mergesort with corrected bounds */
    for(m = 1; m < stop - start; m += m) {
        int step = m << 1;
        for(i = start; i < stop; i += step) {
            from = i;
            mid  = i + m - 1;
            to   = i + step - 1;
            if (mid >= stop)
                break;
            if (to >= stop)
                to = stop - 1;
            merge(a, from, mid, to);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    sort_merge_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
