#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../sort.h"
#include <string.h>

static double sort_merge_kernel_time_acc = 0.0;

void reset_sort_merge_kernel_time(void) { sort_merge_kernel_time_acc = 0.0; }
double get_sort_merge_kernel_time(void) { return sort_merge_kernel_time_acc; }

void merge(TYPE a[SIZE], int start, int m, int stop){
    TYPE temp[SIZE];
    int i, j, k;

    int len1 = m - start + 1;
    int len2 = stop - m;
    
    for(i=0; i<len1; i++){
        temp[i] = a[start + i];
    }

    for(j=0; j<len2; j++){
        temp[len1 + j] = a[m + 1 + j];
    }

    i = 0;
    j = len1 + len2 - 1;

    for(k=start; k<=stop; k++){
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

    for(m=1; m<stop-start; m+=m) {
        int num_merges = (stop - start + 2*m - 1) / (2*m);
        
        #ifdef _OPENMP
        if(m >= 64) {
            #pragma omp parallel for schedule(dynamic, 1) private(from, mid, to)
            for(i=0; i<num_merges; i++) {
                from = start + i * 2 * m;
                mid = from + m - 1;
                to = from + 2 * m - 1;
                
                if(from >= stop) continue;
                
                if(mid >= stop) {
                    mid = stop - 1;
                }
                
                if(to >= stop) {
                    to = stop - 1;
                }
                
                if(from < stop && mid < stop) {
                    merge(a, from, mid, to);
                }
            }
        } else
        #endif
        {
            for(i=start; i<stop; i+=m+m) {
                from = i;
                mid = i+m-1;
                to = i+m+m-1;
                if(to < stop){
                    merge(a, from, mid, to);
                }
                else{
                    merge(a, from, mid, stop-1);
                }
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    sort_merge_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
