#include <time.h>
#include "../sort.h"

static double sort_merge_kernel_time_acc = 0.0;

void reset_sort_merge_kernel_time(void) { sort_merge_kernel_time_acc = 0.0; }
double get_sort_merge_kernel_time(void) { return sort_merge_kernel_time_acc; }

void merge(TYPE a[SIZE], int start, int m, int stop) {
    TYPE temp[SIZE];

    int n_left  = m - start + 1;
    int n_right = stop - m;

    // Copy left half contiguously
    for (int i = 0; i < n_left; ++i) {
        temp[i] = a[start + i];
    }

    // Copy right half contiguously, reversed, directly after left half
    for (int j = 0; j < n_right; ++j) {
        temp[n_left + j] = a[stop - j];
    }

    int i = 0;                  // index into left half in temp
    int j = n_left + n_right-1; // index into right half in temp
    int k = start;

    // Merge
    for (; k <= stop; ++k) {
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
    int start = 0;
    int stop  = SIZE;
    int m, i, from, mid, to;

    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    for (m = 1; m < stop - start; m += m) {
        for (i = start; i < stop; i += m + m) {
            from = i;
            mid  = i + m - 1;
            to   = i + m + m - 1;
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
