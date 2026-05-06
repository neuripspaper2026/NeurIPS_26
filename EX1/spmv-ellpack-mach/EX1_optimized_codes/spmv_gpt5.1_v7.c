#include <time.h>
#include "../spmv.h"

static double spmv_ellpack_kernel_time_acc = 0.0;

void reset_spmv_ellpack_kernel_time(void) { spmv_ellpack_kernel_time_acc = 0.0; }
double get_spmv_ellpack_kernel_time(void) { return spmv_ellpack_kernel_time_acc; }

void ellpack(TYPE nzval[N*L], int32_t cols[N*L], TYPE vec[N], TYPE out[N])
{
    int i, j;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    for (i = 0; i < N; i++) {
        const int base = i * L;
        TYPE sum = out[i];
        for (j = 0; j < L; j++) {
            const int idx = base + j;
            sum += nzval[idx] * vec[cols[idx]];
        }
        out[i] = sum;
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    spmv_ellpack_kernel_time_acc += (double)(kernel_end.tv_sec - kernel_start.tv_sec)
                                  + (double)(kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
