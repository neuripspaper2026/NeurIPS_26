#include <time.h>
#include "../sort.h"

static double sort_merge_kernel_time_acc = 0.0;

void reset_sort_merge_kernel_time(void) { sort_merge_kernel_time_acc = 0.0; }
double get_sort_merge_kernel_time(void) { return sort_merge_kernel_time_acc; }

void merge(TYPE a[SIZE], int start, int m, int stop){
    TYPE temp[SIZE];
    int i, j, k;

    /* Copy left part */
    for(i = start; i <= m; ++i){
        temp[i] = a[i];
    }

    /* Copy right part in reverse order */
    for(j = m + 1; j <= stop; ++j){
        temp[m + 1 + stop - j] = a[j];
    }

    i = start;
    j = stop;

    /* Merge back to a[] using already loaded temps when beneficial */
    for(k = start; k <= stop; ++k){
        TYPE tmp_i = temp[i];
        TYPE tmp_j = temp[j];
        if(tmp_j < tmp_i) {
            a[k] = tmp_j;
            --j;
        } else {
            a[k] = tmp_i;
            ++i;
        }
    }
}

void ms_mergesort(TYPE a[SIZE]) {
    int start = 0;
    int stop = SIZE;
    int i, m, from, mid, to;

    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    for(m = 1; m < stop - start; m += m) {
        int step = m << 1;
        for(i = start; i < stop; i += step) {
            from = i;
            mid  = i + m - 1;
            to   = i + step - 1;
            if(to < stop){
                merge(a, from, mid, to);
            } else {
                merge(a, from, mid, stop - 1);
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    sort_merge_kernel_time_acc +=
        (double)(kernel_end.tv_sec - kernel_start.tv_sec) +
        (double)(kernel_end.tv_nsec - kernel_start.tv_nsec) * 1e-9;
}
