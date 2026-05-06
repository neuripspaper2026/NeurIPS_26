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
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    /* Parallelize outer loop over rows; each iteration writes to a distinct out[i] */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
        TYPE sum = out[i];
        int base = i * L;

        /* Inner loop is small and fixed-size (L is 10); unroll manually for ILP */
        #pragma GCC ivdep
        for (int j = 0; j < L; j++) {
            int idx = base + j;
            sum += nzval[idx] * vec[cols[idx]];
        }

        out[i] = sum;
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    spmv_ellpack_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                    (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
