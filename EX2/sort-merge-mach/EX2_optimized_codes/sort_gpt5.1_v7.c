#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../sort.h"

static double sort_merge_kernel_time_acc = 0.0;

void reset_sort_merge_kernel_time(void) { sort_merge_kernel_time_acc = 0.0; }
double get_sort_merge_kernel_time(void) { return sort_merge_kernel_time_acc; }

void merge(TYPE a[SIZE], int start, int m, int stop){
    /* Use a single temporary buffer sized exactly to the merge range.
       This avoids repeatedly allocating a large stack array and keeps
       accesses contiguous for better cache behavior. */
    const int len = stop - start + 1;
    TYPE temp[len];
    int i, j, k;

    /* Copy left half in-order */
    for(i = 0; i <= m - start; i++){
        temp[i] = a[start + i];
    }

    /* Copy right half in reverse order */
    for(j = m + 1; j <= stop; j++){
        temp[(m - start) + 1 + (stop - j)] = a[j];
    }

    i = 0;
    j = len - 1;

    /* Merge from temp back into a[start..stop] */
    for(k = start; k <= stop; k++){
        TYPE tmp_j = temp[j];
        TYPE tmp_i = temp[i];
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

    /* Bottom-up iterative mergesort.
       Inner loop is parallelized when OpenMP is available and when the
       current run size is small enough to expose useful parallelism. */
    for(m = 1; m < stop - start; m += m) {
#ifdef _OPENMP
        /* Parallelize across independent merge operations for this run size. */
#pragma omp parallel for private(from, mid, to, i) schedule(static)
#endif
        for(i = start; i < stop; i += m + m) {
            from = i;
            mid = i + m - 1;
            to = i + m + m - 1;

            if(mid >= stop - 1) {
                /* Nothing to merge: right run is empty or out-of-range. */
                continue;
            }

            if(to < stop){
                merge(a, from, mid, to);
            } else {
                merge(a, from, mid, stop - 1);
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    sort_merge_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
