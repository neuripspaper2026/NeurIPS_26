#include <time.h>
#include "../sort.h"

static double sort_merge_kernel_time_acc = 0.0;

void reset_sort_merge_kernel_time(void) { sort_merge_kernel_time_acc = 0.0; }
double get_sort_merge_kernel_time(void) { return sort_merge_kernel_time_acc; }

void merge(TYPE a[SIZE], int start, int m, int stop){
    /* Use a single temporary buffer and tighter loop bounds to reduce work */
    TYPE temp[SIZE];
    int left_len  = m - start + 1;
    int right_len = stop - m;
    int i, j, k;

    /* Copy left half directly */
    for (i = 0; i < left_len; ++i) {
        temp[i] = a[start + i];
    }

    /* Copy right half in reverse order starting from the end of temp */
    for (j = 0; j < right_len; ++j) {
        temp[left_len + j] = a[stop - j];
    }

    /* Merge from temp back into a */
    i = 0;                    /* index into left part in temp  */
    j = left_len + right_len - 1; /* index into right part in temp (reverse) */

    for (k = start; k <= stop; ++k) {
        TYPE tmp_j = temp[j];
        TYPE tmp_i = temp[i];
        if (tmp_j < tmp_i) {
            a[k] = tmp_j;
            --j;
        } else {
            a[k] = tmp_i;
            ++i;
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

    for (m = 1; m < stop - start; m += m) {
        for (i = start; i < stop; i += m + m) {
            from = i;
            mid  = i + m - 1;
            to   = i + m + m - 1;
            if (to < stop - 1) {
                merge(a, from, mid, to);
            } else {
                merge(a, from, mid, stop - 1);
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    sort_merge_kernel_time_acc +=
        (kernel_end.tv_sec - kernel_start.tv_sec) +
        (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
