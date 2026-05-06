#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../spmv.h"

static double spmv_crs_kernel_time_acc = 0.0;

void reset_spmv_crs_kernel_time(void) { spmv_crs_kernel_time_acc = 0.0; }
double get_spmv_crs_kernel_time(void) { return spmv_crs_kernel_time_acc; }

void spmv(TYPE val[NNZ], int32_t cols[NNZ], int32_t rowDelimiters[N+1], TYPE vec[N], TYPE out[N]){
    int i, j;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 16) private(j)
    #endif
    for(i = 0; i < N; i++){
        TYPE sum = 0.0;
        int tmp_begin = rowDelimiters[i];
        int tmp_end = rowDelimiters[i+1];
        
        // Manual loop unrolling with 4-way unroll
        int j_end = tmp_begin + ((tmp_end - tmp_begin) / 4) * 4;
        
        for (j = tmp_begin; j < j_end; j += 4){
            TYPE v0 = val[j] * vec[cols[j]];
            TYPE v1 = val[j+1] * vec[cols[j+1]];
            TYPE v2 = val[j+2] * vec[cols[j+2]];
            TYPE v3 = val[j+3] * vec[cols[j+3]];
            sum += v0 + v1 + v2 + v3;
        }
        
        // Handle remaining elements
        for (j = j_end; j < tmp_end; j++){
            sum += val[j] * vec[cols[j]];
        }
        
        out[i] = sum;
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    spmv_crs_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}


