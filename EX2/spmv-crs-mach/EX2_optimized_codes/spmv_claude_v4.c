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

    #pragma omp parallel for schedule(dynamic, 8) private(j)
    spmv_1 : for(i = 0; i < N; i++){
        TYPE sum = 0;
        int tmp_begin = rowDelimiters[i];
        int tmp_end = rowDelimiters[i+1];
        
        // Unroll inner loop by 4 for better instruction-level parallelism
        int j_end_unroll = tmp_begin + ((tmp_end - tmp_begin) / 4) * 4;
        
        spmv_2 : for (j = tmp_begin; j < j_end_unroll; j += 4){
            TYPE Si0 = val[j] * vec[cols[j]];
            TYPE Si1 = val[j+1] * vec[cols[j+1]];
            TYPE Si2 = val[j+2] * vec[cols[j+2]];
            TYPE Si3 = val[j+3] * vec[cols[j+3]];
            sum += Si0 + Si1 + Si2 + Si3;
        }
        
        // Handle remaining elements
        for (j = j_end_unroll; j < tmp_end; j++){
            sum += val[j] * vec[cols[j]];
        }
        
        out[i] = sum;
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    spmv_crs_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}


