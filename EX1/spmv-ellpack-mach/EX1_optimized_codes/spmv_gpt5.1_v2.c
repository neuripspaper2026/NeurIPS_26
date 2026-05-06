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
        const TYPE *restrict nz_row   = &nzval[i * L];
        const int32_t *restrict col_r = &cols[i * L];

        TYPE sum = out[i];

        /* Manually unroll inner loop for fixed L = 10 */
        sum += nz_row[0] * vec[col_r[0]];
        sum += nz_row[1] * vec[col_r[1]];
        sum += nz_row[2] * vec[col_r[2]];
        sum += nz_row[3] * vec[col_r[3]];
        sum += nz_row[4] * vec[col_r[4]];
        sum += nz_row[5] * vec[col_r[5]];
        sum += nz_row[6] * vec[col_r[6]];
        sum += nz_row[7] * vec[col_r[7]];
        sum += nz_row[8] * vec[col_r[8]];
        sum += nz_row[9] * vec[col_r[9]];

        out[i] = sum;
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    spmv_ellpack_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                    (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
