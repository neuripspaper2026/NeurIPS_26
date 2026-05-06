#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../sort.h"

static double sort_merge_kernel_time_acc = 0.0;

void reset_sort_merge_kernel_time(void) { sort_merge_kernel_time_acc = 0.0; }
double get_sort_merge_kernel_time(void) { return sort_merge_kernel_time_acc; }

/* Use a single global temporary buffer to avoid repeated allocations on stack */
static TYPE merge_temp[SIZE];

void merge(TYPE a[SIZE], int start, int m, int stop){
    int i, j, k;

    /* Copy left half directly */
    merge_label1 : for(i = start; i <= m; i++){
        merge_temp[i] = a[i];
    }

    /* Copy right half in reverse order */
    merge_label2 : for(j = m + 1; j <= stop; j++){
        merge_temp[m + 1 + stop - j] = a[j];
    }

    i = start;
    j = stop;

    merge_label3 : for(k = start; k <= stop; k++){
        TYPE tmp_j = merge_temp[j];
        TYPE tmp_i = merge_temp[i];
        if (tmp_j < tmp_i) {
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

    /* Bottom-up mergesort; each level is parallelized when possible */
    mergesort_label1 : for(m = 1; m < stop - start; m += m) {
        int chunk = 2 * m;

        /* Parallelize outer per-run loop; each run works on disjoint segments */
        #ifdef _OPENMP
        #pragma omp parallel for private(from, mid, to, i) schedule(static)
        #endif
        mergesort_label2 : for(i = start; i < stop; i += chunk) {
            from = i;
            mid  = i + m - 1;
            to   = i + chunk - 1;

            if (mid >= stop - 1) {
                /* If the middle index is beyond or at the last element, no valid pair to merge */
                continue;
            }

            if (to < stop) {
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
