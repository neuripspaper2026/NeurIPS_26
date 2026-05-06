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
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    /* Parallel outer loop; avoid race on out[] by giving each i to a single thread */
#ifdef _OPENMP
#pragma omp parallel for private(j) schedule(static)
#endif
    for (i = 0; i < N; i++) {
        TYPE sum = out[i];
        /* Inner loop is very small (L=10); unroll for better ILP and reduced overhead */
#pragma GCC ivdep
        for (j = 0; j < L; j++) {
            const int idx = j + i * L;
            sum += nzval[idx] * vec[cols[idx]];
        }
        out[i] = sum;
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    spmv_ellpack_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                    (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
