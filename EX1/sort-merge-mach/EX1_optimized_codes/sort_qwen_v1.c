#include <time.h>
#include "../sort.h"

static double sort_merge_kernel_time_acc = 0.0;

void reset_sort_merge_kernel_time(void) { sort_merge_kernel_time_acc = 0.0; }
double get_sort_merge_kernel_time(void) { return sort_merge_kernel_time_acc; }

void merge(TYPE a[SIZE], int start, int m, int stop){
    TYPE temp[SIZE];
    int i, j, k;

    // Unroll small loops and use pointer arithmetic for better performance
    TYPE * restrict temp_ptr = temp;
    TYPE * restrict a_ptr = a;
    
    // Copy first half directly
    for(i = start; i <= m; i++){
        temp_ptr[i] = a_ptr[i];
    }

    // Copy second half in reverse order
    for(j = m+1; j <= stop; j++){
        temp_ptr[m+1+stop-j] = a_ptr[j];
    }

    i = start;
    j = stop;

    // Optimize the merge loop by reducing memory accesses
    for(k = start; k <= stop; k++){
        TYPE tmp_j = temp_ptr[j];
        TYPE tmp_i = temp_ptr[i];
        if(tmp_j < tmp_i) {
            a_ptr[k] = tmp_j;
            j--;
        } else {
            a_ptr[k] = tmp_i;
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

    // Optimize outer loops: pre-calculate bounds where possible
    for(m = 1; m < SIZE; m += m) {
        // Reduce condition checks in inner loop
        int limit = SIZE - m;
        for(i = start; i < limit; i += m + m) {
            from = i;
            mid = i + m - 1;
            to = i + m + m - 1;
            // Simplify condition by pre-calculating
            if(to < SIZE){
                merge(a, from, mid, to);
            }
            else{
                merge(a, from, mid, SIZE);
            }
        }
        // Handle remaining elements if any
        if (i < SIZE) {
            from = i;
            mid = i + m - 1;
            if (mid >= SIZE) mid = SIZE - 1;
            merge(a, from, mid, SIZE);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    sort_merge_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
