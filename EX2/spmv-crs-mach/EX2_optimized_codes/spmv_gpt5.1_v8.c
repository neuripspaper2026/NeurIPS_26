#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../spmv.h"

static double spmv_crs_kernel_time_acc = 0.0;

void reset_spmv_crs_kernel_time(void) { spmv_crs_kernel_time_acc = 0.0; }
double get_spmv_crs_kernel_time(void) { return spmv_crs_kernel_time_acc; }

void spmv(TYPE val[NNZ], int32_t cols[NNZ], int32_t rowDelimiters[N+1], TYPE vec[N], TYPE out[N]){
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    /* Parallelize outer loop by rows; each row writes to a unique out[i] */
    #ifdef _OPENMP
    #pragma omp parallel
    {
        int i, j;
        TYPE sum, Si;

        #pragma omp for schedule(static)
        for(i = 0; i < N; i++){
            sum = 0.0;
            int tmp_begin = rowDelimiters[i];
            int tmp_end   = rowDelimiters[i+1];

            /* Inner loop kept serial per row; simple reduction pattern */
            for (j = tmp_begin; j < tmp_end; j++){
                Si = val[j] * vec[cols[j]];
                sum += Si;
            }
            out[i] = sum;
        }
    }
    #else
    {
        int i, j;
        TYPE sum, Si;

        for(i = 0; i < N; i++){
            sum = 0.0;
            int tmp_begin = rowDelimiters[i];
            int tmp_end   = rowDelimiters[i+1];

            for (j = tmp_begin; j < tmp_end; j++){
                Si = val[j] * vec[cols[j]];
                sum += Si;
            }
            out[i] = sum;
        }
    }
    #endif

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    spmv_crs_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}


