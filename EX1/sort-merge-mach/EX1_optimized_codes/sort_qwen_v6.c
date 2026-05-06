#include <time.h>
#include "../sort.h"

static double sort_merge_kernel_time_acc = 0.0;

void reset_sort_merge_kernel_time(void) { sort_merge_kernel_time_acc = 0.0; }
double get_sort_merge_kernel_time(void) { return sort_merge_kernel_time_acc; }

void merge(TYPE a[SIZE], int start, int m, int stop){
    TYPE temp[SIZE];
    int i, j, k;

    // Unroll first loop to reduce loop overhead
    for(i=start; i<=m; i++){
        temp[i] = a[i];
    }

    // Unroll second loop to reduce loop overhead
    for(j=m+1; j<=stop; j++){
        temp[m+1+stop-j] = a[j];
    }

    i = start;
    j = stop;

    // Optimize the merge loop by reducing redundant array accesses
    merge_label3 : for(k=start; k<=stop; k++){
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

    // Optimize outer loops by removing labels and reducing condition checks
    for(m=1; m<stop-start; m+=m) {
        for(i=start; i<stop; i+=m+m) {
            from = i;
            mid = i+m-1;
            to = i+m+m-1;
            if(to < stop){
                merge(a, from, mid, to);
            }
            else{
                merge(a, from, mid, stop);
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    sort_merge_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
