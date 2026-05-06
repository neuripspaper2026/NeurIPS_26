#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../spmv.h"

static double spmv_crs_kernel_time_acc = 0.0;

void reset_spmv_crs_kernel_time(void) { spmv_crs_kernel_time_acc = 0.0; }
double get_spmv_crs_kernel_time(void) { return spmv_crs_kernel_time_acc; }

void spmv(TYPE val[NNZ], int32_t cols[NNZ], int32_t rowDelimiters[N+1], TYPE vec[N], TYPE out[N]){
    int i;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    /* Parallelize outer loop over rows; each iteration writes to distinct out[i] */
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(val, cols, rowDelimiters, vec, out) private(i)
#endif
    for(i = 0; i < N; i++){
        TYPE sum = 0.0;
        int tmp_begin = rowDelimiters[i];
        int tmp_end   = rowDelimiters[i+1];

        /* Inner loop is completely independent, good candidate for vectorization */
#pragma omp simd reduction(+:sum)
        for (int j = tmp_begin; j < tmp_end; j++){
            sum += val[j] * vec[cols[j]];
        }
        out[i] = sum;
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    spmv_crs_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}


