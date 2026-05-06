#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../sort.h"

static double sort_merge_kernel_time_acc = 0.0;

void reset_sort_merge_kernel_time(void) { sort_merge_kernel_time_acc = 0.0; }
double get_sort_merge_kernel_time(void) { return sort_merge_kernel_time_acc; }

/* Use a single global temporary buffer to avoid repeated stack allocations
 * and large per-thread stack use. This is safe on the given fixed SIZE. */
static TYPE temp_global[SIZE];

void merge(TYPE a[SIZE], int start, int m, int stop){
    TYPE *restrict temp = temp_global;
    int i, j, k;

    /* Copy left half as-is */
    merge_label1 : for(i = start; i <= m; i++){
        temp[i] = a[i];
    }

    /* Copy right half in reverse order */
    merge_label2 : for(j = m + 1; j <= stop; j++){
        temp[m + 1 + stop - j] = a[j];
    }

    i = start;
    j = stop;

    /* Merge from temp back into a */
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
     * Parallelize the outer loop over segments of size 2*m.
     * Each iteration of the inner loop works on a disjoint range of 'a',
     * so it is safe to parallelize without synchronization. */
    mergesort_label1 : for(m = 1; m < stop - start; m += m) {
#ifdef _OPENMP
        /* Parallelize over blocks; collapse is not needed, as 'm' is fixed
         * per outer iteration. Dynamic scheduling not required; static
         * is fine due to roughly uniform work. */
#pragma omp parallel for private(from, mid, to, i) schedule(static)
#endif
        mergesort_label2 : for(i = start; i < stop; i += m + m) {
            from = i;
            mid  = i + m - 1;
            to   = i + m + m - 1;

            if (mid >= stop) {
                /* Nothing to merge: right half doesn't exist */
                continue;
            }

            if (to < stop - 1) {
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
