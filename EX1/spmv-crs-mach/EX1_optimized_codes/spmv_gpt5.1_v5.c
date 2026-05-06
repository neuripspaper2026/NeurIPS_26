#include <time.h>
#include "../spmv.h"

static double spmv_crs_kernel_time_acc = 0.0;

void reset_spmv_crs_kernel_time(void) { spmv_crs_kernel_time_acc = 0.0; }
double get_spmv_crs_kernel_time(void) { return spmv_crs_kernel_time_acc; }

void spmv(TYPE val[NNZ], int32_t cols[NNZ], int32_t rowDelimiters[N+1], TYPE vec[N], TYPE out[N]){
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    for (int i = 0; i < N; i++) {
        TYPE sum = 0.0;
        int tmp_begin = rowDelimiters[i];
        int tmp_end   = rowDelimiters[i+1];

        int j = tmp_begin;
        int limit = tmp_end - ((tmp_end - tmp_begin) & 1);

        for (; j < limit; j += 2) {
            int c0 = cols[j];
            int c1 = cols[j + 1];

            TYPE v0 = val[j];
            TYPE v1 = val[j + 1];

            sum += v0 * vec[c0] + v1 * vec[c1];
        }

        for (; j < tmp_end; j++) {
            sum += val[j] * vec[cols[j]];
        }

        out[i] = sum;
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    spmv_crs_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}


