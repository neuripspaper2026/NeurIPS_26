#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../sort.h"

static double sort_merge_kernel_time_acc = 0.0;

void reset_sort_merge_kernel_time(void) { sort_merge_kernel_time_acc = 0.0; }
double get_sort_merge_kernel_time(void) { return sort_merge_kernel_time_acc; }

void merge(TYPE a[SIZE], int start, int m, int stop){
    TYPE temp[SIZE];
    int i, j, k;

    /* Copy left half directly */
    merge_label1 : for(i = start; i <= m; i++){
        temp[i] = a[i];
    }

    /* Copy right half in reverse order into temp to enable bidirectional merge */
    merge_label2 : for(j = m + 1; j <= stop; j++){
        temp[m + 1 + stop - j] = a[j];
    }

    i = start;
    j = stop;

    /* Merge by always selecting min of current left/right candidates */
    merge_label3 : for(k = start; k <= stop; k++){
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
       Parallelize the outer merge pass when the subproblem size m
       is large enough to amortize threading overhead. */
    mergesort_label1 : for(m = 1; m < stop - start; m += m) {
        /* For very small m, the work per task is tiny; keep it serial.
           Threshold chosen small due to SIZE=2048 but still avoids
           parallel overhead on trivial passes. */
        if (m < 16) {
            mergesort_label2 : for(i = start; i < stop; i += m + m) {
                from = i;
                mid  = i + m - 1;
                to   = i + m + m - 1;
                if (to < stop) {
                    merge(a, from, mid, to);
                } else {
                    merge(a, from, mid, stop - 1);
                }
            }
        } else {
#ifdef _OPENMP
#pragma omp parallel for private(from, mid, to) schedule(static)
#endif
            mergesort_label2 : for(i = start; i < stop; i += m + m) {
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
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    sort_merge_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
