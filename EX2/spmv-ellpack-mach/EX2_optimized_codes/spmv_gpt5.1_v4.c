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
    int i, j;
    TYPE Si;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    /* Parallelize outer loop; each iteration writes a distinct out[i] */
#ifdef _OPENMP
#pragma omp parallel for private(j, Si) schedule(static)
#endif
    for (i = 0; i < N; i++) {
        TYPE sum = out[i];

        /* Unroll inner loop since L is a small fixed constant (10) */
#pragma GCC ivdep
        for (j = 0; j < L; j += 2) {
            int idx0 = j + i * L;
            int idx1 = idx0 + 1;

            TYPE v0 = nzval[idx0];
            TYPE v1 = nzval[idx1];

            int32_t c0 = cols[idx0];
            int32_t c1 = cols[idx1];

            TYPE x0 = vec[c0];
            TYPE x1 = vec[c1];

            sum += v0 * x0 + v1 * x1;
        }

        out[i] = sum;
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    spmv_ellpack_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                    (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
