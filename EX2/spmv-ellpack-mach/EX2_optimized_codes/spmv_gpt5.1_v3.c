#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../spmv.h"

static double spmv_ellpack_kernel_time_acc = 0.0;

void reset_spmv_ellpack_kernel_time(void) { spmv_ellpack_kernel_time_acc = 0.0; }
double get_spmv_ellpack_kernel_time(void) { return spmv_ellpack_kernel_time_acc; }

void ellpack(TYPE nzval[N*L], int32_t cols[N*L], TYPE vec[N], TYPE out[N])
{
    int i;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    /* Parallelize outer loop over rows; use static scheduling for balanced work.
       Keep sum as a local variable to enable better optimization and vectorization. */
#ifdef _OPENMP
#pragma omp parallel for private(i) schedule(static)
#endif
    for (i = 0; i < N; i++) {
        TYPE sum = out[i];
        TYPE *restrict nz_row = &nzval[i * L];
        int32_t *restrict col_row = &cols[i * L];

        /* Manually unroll inner loop for better ILP and vectorization hints. */
        int j = 0;
        int limit = L & ~3; /* largest multiple of 4 <= L */

        for (; j < limit; j += 4) {
            sum += nz_row[j    ] * vec[col_row[j    ]];
            sum += nz_row[j + 1] * vec[col_row[j + 1]];
            sum += nz_row[j + 2] * vec[col_row[j + 2]];
            sum += nz_row[j + 3] * vec[col_row[j + 3]];
        }
        for (; j < L; j++) {
            sum += nz_row[j] * vec[col_row[j]];
        }

        out[i] = sum;
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    spmv_ellpack_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                    (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
