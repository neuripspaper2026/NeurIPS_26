#include <time.h>
#include "../spmv.h"

static double spmv_ellpack_kernel_time_acc = 0.0;

void reset_spmv_ellpack_kernel_time(void) { spmv_ellpack_kernel_time_acc = 0.0; }
double get_spmv_ellpack_kernel_time(void) { return spmv_ellpack_kernel_time_acc; }

void ellpack(TYPE nzval[N*L], int32_t cols[N*L], TYPE vec[N], TYPE out[N])
{
    int i;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    for (i = 0; i < N; i++) {
        TYPE sum = out[i];

        TYPE v0 = vec[cols[i*L + 0]];
        TYPE v1 = vec[cols[i*L + 1]];
        TYPE v2 = vec[cols[i*L + 2]];
        TYPE v3 = vec[cols[i*L + 3]];
        TYPE v4 = vec[cols[i*L + 4]];
        TYPE v5 = vec[cols[i*L + 5]];
        TYPE v6 = vec[cols[i*L + 6]];
        TYPE v7 = vec[cols[i*L + 7]];
        TYPE v8 = vec[cols[i*L + 8]];
        TYPE v9 = vec[cols[i*L + 9]];

        const TYPE *restrict nz_row = &nzval[i*L];

        sum += nz_row[0] * v0;
        sum += nz_row[1] * v1;
        sum += nz_row[2] * v2;
        sum += nz_row[3] * v3;
        sum += nz_row[4] * v4;
        sum += nz_row[5] * v5;
        sum += nz_row[6] * v6;
        sum += nz_row[7] * v7;
        sum += nz_row[8] * v8;
        sum += nz_row[9] * v9;

        out[i] = sum;
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    spmv_ellpack_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                    (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
