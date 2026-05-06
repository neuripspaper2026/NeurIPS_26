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

    merge_label1 : for(i=0; i<len1; i++){
        temp[i] = a[start + i];
    }

    merge_label2 : for(j=0; j<len2; j++){
        temp[len1 + j] = a[m + 1 + j];
    }

    i = 0;
    j = len1 + len2 - 1;
    k = start;

    merge_label3 : for(int idx=0; idx<=stop-start; idx++){
        TYPE tmp_j = temp[j];
        TYPE tmp_i = temp[i];
        if(tmp_j < tmp_i) {
            a[k] = tmp_j;
            j--;
        } else {
            a[k] = tmp_i;
            i++;
        }
        k++;
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
            to = i+m+m-1;
            if(to < stop-1){
                merge(a, from, mid, to);
            }
            else if(mid < stop-1){
                merge(a, from, mid, stop-1);
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    sort_merge_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
