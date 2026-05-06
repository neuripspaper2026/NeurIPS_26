#include <time.h>
#include "../sort.h"

static double sort_merge_kernel_time_acc = 0.0;

void reset_sort_merge_kernel_time(void) { sort_merge_kernel_time_acc = 0.0; }
double get_sort_merge_kernel_time(void) { return sort_merge_kernel_time_acc; }

void merge(TYPE a[SIZE], int start, int m, int stop) {
    TYPE temp[SIZE];
    int i = start;
    int j = stop;
    int k;

    /* Copy left part */
    for (; i <= m; ++i) {
        temp[i] = a[i];
    }

    /* Copy right part in reverse */
    for (int jj = m + 1; jj <= stop; ++jj) {
        temp[m + 1 + stop - jj] = a[jj];
    }

    i = start;
    j = stop;

    /* Merge */
    for (k = start; k <= stop; ++k) {
        TYPE tmp_i = temp[i];
        TYPE tmp_j = temp[j];
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
        int step = m + m;
        for (i = start; i < stop; i += step) {
            from = i;
            mid  = i + m - 1;
            to   = i + step - 1;
            if (to < stop) {
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
