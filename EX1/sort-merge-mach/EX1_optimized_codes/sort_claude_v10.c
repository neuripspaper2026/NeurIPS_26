#include <time.h>
#include <string.h>
#include "../sort.h"

static double sort_merge_kernel_time_acc = 0.0;

void reset_sort_merge_kernel_time(void) { sort_merge_kernel_time_acc = 0.0; }
double get_sort_merge_kernel_time(void) { return sort_merge_kernel_time_acc; }

void merge(TYPE a[SIZE], int start, int m, int stop){
    TYPE temp[SIZE];
    int i, j, k;
    int len1 = m - start + 1;
    int len2 = stop - m;

    memcpy(&temp[start], &a[start], len1 * sizeof(TYPE));
    
    merge_label2 : for(j=m+1; j<=stop; j++){
        temp[m+1+stop-j] = a[j];
    }

    i = start;
    j = stop;

    merge_label3 : for(k=start; k<=stop; k++){
        TYPE tmp_j = temp[j];
        TYPE tmp_i = temp[i];
        int cmp = tmp_j < tmp_i;
        a[k] = cmp ? tmp_j : tmp_i;
        j -= cmp;
        i += !cmp;
    }
}

void ms_mergesort(TYPE a[SIZE]) {
    int start, stop;
    int i, m, from, mid, to;

    start = 0;
    stop = SIZE;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    mergesort_label1 : for(m=1; m<stop-start; m<<=1) {
        mergesort_label2 : for(i=start; i<stop; i+=(m<<1)) {
            from = i;
            mid = i+m-1;
            to = i+(m<<1)-1;
            int actual_to = (to < stop) ? to : stop;
            if(mid < stop){
                merge(a, from, mid, actual_to);
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    sort_merge_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
