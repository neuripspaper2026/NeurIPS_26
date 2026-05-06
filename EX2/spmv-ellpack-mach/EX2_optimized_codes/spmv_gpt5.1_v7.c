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

    // Parallelize outer loop; each iteration writes a distinct out[i]
    // Use static scheduling to keep work balanced and cache‑friendly
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        const int base = i * L;
        TYPE sum = out[i];

        // Unroll inner loop for L == 10 to enable better ILP and vectorization
        // and reduce loop control overhead.
        #pragma GCC ivdep
        for (int j = 0; j < L; j += 2) {
            int idx0 = base + j;
            int idx1 = base + j + 1;

            TYPE v0 = nzval[idx0];
            TYPE v1 = nzval[idx1];

            int c0 = cols[idx0];
            int c1 = cols[idx1];

            sum += v0 * vec[c0];
            sum += v1 * vec[c1];
        }

        out[i] = sum;
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    spmv_ellpack_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                    (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
